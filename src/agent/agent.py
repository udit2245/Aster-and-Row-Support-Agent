import json
import re

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, LLM_MODEL
from src.retrieval.retriever import retrieve
from src.retrieval.policy_selector import (
    build_policy_query,
    select_policies,
)
from src.tools.order_tool import lookup_order
from src.agent.generator import generate_answer


# ---------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------
# Planner Prompt
# ---------------------------------------------------------

PLANNER_PROMPT = """
You are the routing brain of a customer-support AI agent
for Aster & Row.

Your job is to determine what the application should do
with the user's current message.

Available actions:

1. "knowledge"
   Use this for questions about company policies,
   products, shipping, returns, warranties, memberships,
   FAQs, and other general company information.

2. "order"
   Use this when the user is asking about a specific order
   and an order ID is available either in the current message
   or from recent conversation context.

3. "order_and_knowledge"
   Use this when the user asks something about a specific order
   AND the answer requires company policy or knowledge-base
   information.

4. "clarify"
   Use this when important information is genuinely missing
   and the request cannot reasonably be answered.

Important rules:

- Never invent an order ID.
- Use conversation history to resolve references such as
  "my order", "that order", "it", or "the order".
- If an order ID appears in the current user message,
  prefer that order ID.
- If the user refers to an order from previous conversation,
  reuse the previously established order ID.
- Do not assume an order ID if no evidence exists.
- Return ONLY valid JSON.

Required JSON format:

{
    "action": "knowledge | order | order_and_knowledge | clarify",
    "order_id": "ORD-XXXX or null",
    "search_query": "knowledge-base search query or null",
    "reason": "short explanation"
}
"""


# ---------------------------------------------------------
# Order ID Normalization
# ---------------------------------------------------------

def normalize_order_id(order_id: str | None) -> str | None:
    """
    Convert harmless order-ID variations into the canonical
    format used by the dataset: ORD-<digits>.

    Examples:
        "ORD-1001" -> "ORD-1001"
        "ord-1001" -> "ORD-1001"
        "ORD1001"  -> "ORD-1001"
        "1001"     -> "ORD-1001"
        " 1001 "   -> "ORD-1001"

    Returns None for empty or malformed values.
    """

    if order_id is None:
        return None

    value = str(order_id).strip().upper()

    if not value:
        return None

    # Canonical form: ORD-1234
    match = re.fullmatch(
        r"ORD-(\d+)",
        value
    )

    if match:
        return f"ORD-{match.group(1)}"

    # Harmless variation: ORD1234
    match = re.fullmatch(
        r"ORD(\d+)",
        value
    )

    if match:
        return f"ORD-{match.group(1)}"

    # Numeric-only form: 1234
    match = re.fullmatch(
        r"\d+",
        value
    )

    if match:
        return f"ORD-{match.group(0)}"

    return None


# ---------------------------------------------------------
# Order ID Extraction
# ---------------------------------------------------------

def extract_order_id(
    text: str,
    history: list[dict] | None = None
) -> str | None:
    """
    Extract an order ID from the current user message.

    Supported forms:

        ORD-1007
        ORD1007
        order 1007
        order ID 1007
        order number 1007
        1007  (only when recent conversation context
               clearly establishes that an order is being
               discussed)

    Numeric-only values are NOT blindly treated as order IDs.
    They require order-related context.
    """

    text_upper = text.upper().strip()

    # -----------------------------------------------------
    # 1. Explicit order ID formats
    # -----------------------------------------------------

    match = re.search(
        r"\bORD[-\s]?(\d+)\b",
        text_upper
    )

    if match:
        return normalize_order_id(
            match.group(1)
        )

    # -----------------------------------------------------
    # 2. Current message explicitly mentions an order
    # -----------------------------------------------------

    if re.search(
        r"\b(order|order\s+id|order\s+number|ord)\b",
        text_upper
    ):

        match = re.search(
            r"\b(\d{3,})\b",
            text_upper
        )

        if match:
            return normalize_order_id(
                match.group(1)
            )

    # -----------------------------------------------------
    # 3. Numeric-only follow-up
    #
    # Example:
    #
    # User: Where is my order?
    # Agent: Could you provide your order ID?
    # User: 1007
    #
    # The current message contains only "1007", so we
    # inspect recent conversation history to determine
    # whether the number is clearly an order ID.
    # -----------------------------------------------------

    numeric_match = re.fullmatch(
        r"\s*(\d{3,})\s*",
        text
    )

    if numeric_match and history:

        recent_history = history[-6:]

        history_text = "\n".join(
            message.get("content", "")
            for message in recent_history
        ).upper()

        # Look for clear order-related context.
        order_context = re.search(
            r"\b(order|order\s+id|order\s+number|ord)\b",
            history_text
        )

        if order_context:

            return normalize_order_id(
                numeric_match.group(1)
            )

    return None


# ---------------------------------------------------------
# Planner
# ---------------------------------------------------------

def plan(
    question: str,
    history: list[dict]
) -> dict:
    """
    Ask Gemini to determine which capability is required.

    The model receives recent conversation history so that
    follow-up questions can reference previously discussed
    orders or topics.
    """

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-6:]
    )

    if not history_text:
        history_text = "(No previous conversation.)"

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=f"""
CONVERSATION HISTORY:

{history_text}

CURRENT USER QUESTION:

{question}

Return the routing decision as JSON only.
""",
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    decision = json.loads(
        response.text
    )

    # -----------------------------------------------------
    # Application-level validation
    # -----------------------------------------------------

    # Never blindly trust the LLM to invent/extract
    # an order ID.
    #
    # IMPORTANT:
    # Pass conversation history so numeric-only follow-ups
    # such as "1007" can be resolved safely.

    detected_order_id = extract_order_id(
        question,
        history
    )

    if detected_order_id:

        # Explicitly detected ID from the user's message
        # always takes priority over the LLM's value.
        decision["order_id"] = detected_order_id

    else:

        # If Gemini supplied an order ID, normalize it.
        # If it is malformed, this returns None.
        decision["order_id"] = normalize_order_id(
            decision.get("order_id")
        )

    # -----------------------------------------------------
    # Validate action
    # -----------------------------------------------------

    valid_actions = {
        "knowledge",
        "order",
        "order_and_knowledge",
        "clarify",
    }

    if decision.get("action") not in valid_actions:

        decision["action"] = "clarify"

    # -----------------------------------------------------
    # Normalize missing values
    # -----------------------------------------------------

    if not decision.get("order_id"):

        decision["order_id"] = None

    if not decision.get("search_query"):

        decision["search_query"] = None

    return decision


# ---------------------------------------------------------
# Safe Order Context
# ---------------------------------------------------------

def build_order_context(order: dict) -> str:
    """
    Convert the safe result from the order tool into
    trusted context for Gemini.

    Sensitive/internal fields never reach this function.
    """

    return f"""
TRUSTED ORDER TOOL RESULT

Order ID:
{order["order_id"]}

Status:
{order["status"]}

Membership tier:
{order["membership_tier"]}

Placed At:
{order["placed_at"]}

Snapshot At:
{order["snapshot_at"]}

Status Updated At:
{order["status_updated_at"]}

Shipped At:
{order["shipped_at"]}

Delivered At:
{order["delivered_at"]}

Carrier:
{order["carrier"]}

Tracking Number:
{order["tracking_number"]}

Estimated Delivery:
{order["estimated_delivery"]}

Amount available in the order dataset:
{order["amount_available"]}

Customer-safe message:
{order["customer_safe_message"]}

Items:
{json.dumps(order["items"], indent=2)}
"""


# ---------------------------------------------------------
# Agent
# ---------------------------------------------------------

def run_agent(
    question: str,
    history: list[dict]
) -> str:
    """
    Main agent workflow.

    Possible paths:

        knowledge
            -> RAG
            -> Gemini

        order
            -> Order Tool
            -> Gemini

        order_and_knowledge
            -> Order Tool
            -> RAG
            -> Gemini

        clarify
            -> Clarification response
    """

    # -----------------------------------------------------
    # PLAN
    # -----------------------------------------------------

    decision = plan(
        question,
        history
    )

    action = decision.get(
        "action"
    )

    order_id = decision.get(
        "order_id"
    )

    search_query = decision.get(
        "search_query"
    )

    print("\nAGENT DECISION")
    print("=" * 60)

    print(
        f"Action: {action}"
    )

    print(
        f"Order ID: {order_id}"
    )

    print(
        f"Search Query: {search_query}"
    )

    print(
        f"Reason: {decision.get('reason')}"
    )

    # -----------------------------------------------------
    # KNOWLEDGE
    # -----------------------------------------------------

    if action == "knowledge":

        query = (
            search_query
            or question
        )

        documents = retrieve(
            query,
            top_k=8
        )

        documents = select_policies(
            documents,
            query=query,
            order=None,
            max_documents=5
        )

        if not documents:

            return (
                "I don't have enough information in the "
                "knowledge base to answer that reliably. "
                "I recommend contacting a support specialist."
            )

        return generate_answer(
            question,
            documents
        )

    # -----------------------------------------------------
    # ORDER
    # -----------------------------------------------------

    if action == "order":

        if not order_id:

            return (
                "Could you provide your order ID so I can "
                "look up the order?"
            )

        order = lookup_order(
            order_id
        )

        if order is None:

            return (
                f"I couldn't find an order with ID "
                f"{order_id}. Please check the order ID "
                "and try again."
            )

        order_context = build_order_context(
            order
        )

        return generate_answer(
            question,
            [],
            extra_context=order_context
        )

    # -----------------------------------------------------
    # ORDER + KNOWLEDGE
    # -----------------------------------------------------

    if action == "order_and_knowledge":

        if not order_id:

            return (
                "Could you provide your order ID so I can "
                "check the order and the applicable policy?"
            )

        # ---------------------------------------------
        # Step 1: Get order
        # ---------------------------------------------

        order = lookup_order(
            order_id
        )

        if order is None:

            return (
                f"I couldn't find an order with ID "
                f"{order_id}. Please check the order ID "
                "and try again."
            )

        # ---------------------------------------------
        # Step 2: Build order-aware query
        # ---------------------------------------------

        query = build_policy_query(
            question,
            order
        )

        # ---------------------------------------------
        # Step 3: Retrieve broad candidate set
        # ---------------------------------------------

        documents = retrieve(
            query,
            top_k=8
        )

        # ---------------------------------------------
        # Step 4: Apply policy applicability
        # ---------------------------------------------

        documents = select_policies(
            documents,
            query=query,
            order=order,
            max_documents=5
        )

        # ---------------------------------------------
        # Step 5: Build safe order context
        # ---------------------------------------------

        order_context = build_order_context(
            order
        )

        # ---------------------------------------------
        # Step 6: Generate grounded answer
        # ---------------------------------------------

        return generate_answer(
            question,
            documents,
            extra_context=order_context
        )

    # -----------------------------------------------------
    # CLARIFY
    # -----------------------------------------------------

    if action == "clarify":

        return (
            "I need a little more information to help "
            "with that. Could you provide the order ID "
            "or a few more details about your question?"
        )

    # -----------------------------------------------------
    # Unexpected action fallback
    # -----------------------------------------------------

    return (
        "I couldn't determine how to handle that request. "
        "Please provide a little more detail."
    )
