import json
import sys
import io
import re
import time
from pathlib import Path
from contextlib import redirect_stdout

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.agent import run_agent


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

EVALUATION_DIR = ROOT / "evaluation"

VISIBLE_CASES_FILE = EVALUATION_DIR / "visible-cases.json"
CUSTOM_CASES_FILE = EVALUATION_DIR / "custom-cases.json"


# ---------------------------------------------------------
# Gemini/free-tier rate protection
# ---------------------------------------------------------

# Keeps calls below a 5-requests/minute style limit.
MIN_SECONDS_BETWEEN_AGENT_CALLS = 13.5
MAX_429_RETRIES = 3

_last_agent_call = 0.0


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------

def load_cases(path: Path) -> list[dict]:
    """Load evaluation cases from a JSON file."""

    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    cases = data.get("cases", [])

    if not isinstance(cases, list):
        raise ValueError(
            f"{path} must contain a top-level 'cases' list."
        )

    return cases


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text so harmless formatting differences do not
    cause false failures.

    Examples:
        45 calendar days
        45-calendar-days
        **45-calendar-day**

    are treated much more similarly.
    """

    text = str(text)

    replacements = {
        "–": "-",
        "—": "-",
        "‑": "-",
        "−": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "`": "",
        "*": "",
        "_": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_phrase(response: str, phrase: str) -> bool:
    """
    Format-tolerant literal matching.

    This handles:
        45 calendar days
        45-calendar-days
        **45-calendar-day

    without turning arbitrary semantic claims into matches.
    """

    response_n = normalize_text(response)
    phrase_n = normalize_text(phrase)

    # Normal substring match first.
    if phrase_n in response_n:
        return True

    # Then allow punctuation/hyphen/space differences and
    # simple singular/plural differences.
    words = re.findall(r"[a-z0-9]+", phrase_n)

    if not words:
        return False

    pattern_parts = []

    for word in words:

        # "days" can match "day" or "days".
        if word.endswith("s") and len(word) > 3:
            stem = re.escape(word[:-1])
            pattern_parts.append(stem + r"s?")
        else:
            pattern_parts.append(re.escape(word))

    pattern = r"[\s\W_]+".join(pattern_parts)

    return re.search(
        pattern,
        response_n,
        flags=re.IGNORECASE,
    ) is not None


# ---------------------------------------------------------
# Concept-level matching
# ---------------------------------------------------------

# These are intentionally explicit.
#
# must_include_concepts are NOT treated as exact prose.
# They represent behavior/meaning that can legitimately be
# expressed in different ways.
#
# Unknown concepts fall back to literal matching rather than
# being guessed semantically.

CONCEPT_PATTERNS = {

    # -----------------------------------------------------
    # Return windows
    # -----------------------------------------------------

    "45 calendar days": [
        r"\b45\s*[- ]?\s*calendar\s*[- ]?\s*days?\b",
        r"\b45\s*[- ]?\s*day\s+return\s+window\b",
    ],

    "30 calendar days": [
        r"\b30\s*[- ]?\s*calendar\s*[- ]?\s*days?\b",
        r"\b30\s*[- ]?\s*day\s+return\s+window\b",
    ],

    # -----------------------------------------------------
    # International shipping
    # -----------------------------------------------------

    "Canada is supported": [
        r"\bships?\s+(?:internationally\s+)?(?:only\s+)?to\s+canada\b",
        r"\bshipping\s+(?:is\s+)?(?:available|supported)\s+(?:to|in)\s+canada\b",
        r"\bcanada\s+(?:is\s+)?(?:a\s+)?supported\s+(?:country|destination)\b",
        r"\bcanada\s+is\s+(?:available|supported)\b",
    ],

    "shipping to Germany is not currently available": [
        r"\b(?:cannot|can't|can\s+not|unable\s+to)\s+ship\s+to\s+germany\b",
        r"\bshipping\s+to\s+germany\s+(?:is\s+)?(?:not\s+available|unavailable)\b",
        r"\bshipping\s+to\s+other\s+countries\s+is\s+not\s+available\b",
        r"\bonly\s+to\s+canada\b",
        r"\bgermany\s+(?:is\s+)?(?:not\s+supported|unavailable)\b",
    ],

    "5-9 business days after dispatch": [
        r"\b5\s*[- ]?\s*9\s+business\s+days?\s+(?:after|from)\s+dispatch\b",
        r"\b5\s+to\s+9\s+business\s+days?\s+(?:after|from)\s+dispatch\b",
        r"\bdelivery\s+(?:takes?|is\s+estimated\s+at)\s+5\s*[- ]?\s*9\s+business\s+days?\b",
    ],

    "duties or taxes are not prepaid": [
        r"\bdut(?:y|ies)\s+(?:and|or)\s+tax(?:es)?\s+(?:are\s+)?not\s+(?:prepaid|included)\b",
        r"\b(?:duties|taxes|customs\s+(?:duties|charges))\s+(?:are\s+)?(?:the\s+customer'?s\s+responsibility|paid\s+by\s+the\s+customer)\b",
        r"\bimport\s+dut(?:y|ies)\s+and\s+tax(?:es)?\s+(?:are\s+)?(?:not\s+included|payable\s+by\s+the\s+customer)\b",
    ],

    # -----------------------------------------------------
    # Order status
    # -----------------------------------------------------

    # "in transit" is a valid expression of a shipped order
    # for the evaluation case shown in the assignment.
    "shipped": [
        r"\bshipped\b",
        r"\bdispatched\b",
        r"\bin\s+transit\b",
        r"\bon\s+the\s+way\b",
        r"\bhas\s+left\s+(?:our\s+warehouse|the\s+warehouse)\b",
    ],

    # -----------------------------------------------------
    # Final-sale damaged-item exception
    # -----------------------------------------------------

    "final sale does not block damaged-item review": [
        r"\bfinal[- ]sale\b.*\b(?:damaged|defective)\b",
        r"\b(?:damaged|defective)\b.*\bfinal[- ]sale\b",
        r"\bfinal[- ]sale\b.*\b(?:review|exception|claim)\b",
    ],

    "report within 7 days": [
        r"\breport\b.*\bwithin\s+7\s+days?\b",
        r"\bwithin\s+7\s+days?\b.*\breport\b",
    ],

    "human review before approval": [
        r"\bhuman\s+(?:review|approval)\b",
        r"\breviewed?\s+by\s+(?:a\s+)?human\b",
        r"\bsupport\s+(?:specialist|team)\b.*\b(?:review|approve)\b",
    ],
}


def concept_satisfied(response: str, concept: str) -> bool:
    """
    Check a concept without requiring exact wording.

    Strategy:
        1. Flexible literal match.
        2. Explicit semantic patterns for known concepts.
        3. Literal fallback for unknown concepts.

    This prevents the evaluator from becoming an uncontrolled
    second LLM judge.
    """

    if contains_phrase(response, concept):
        return True

    normalized_concept = normalize_text(concept)

    for key, patterns in CONCEPT_PATTERNS.items():

        if normalize_text(key) != normalized_concept:
            continue

        text = normalize_text(response)

        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        )

    return False


# ---------------------------------------------------------
# Required content
# ---------------------------------------------------------

def check_required_content(
    response: str,
    expectations: dict,
) -> tuple[bool, list[str]]:

    failures = []

    # must_include:
    # Literal claims, but formatting-tolerant.
    for phrase in expectations.get(
        "must_include",
        [],
    ):

        if not contains_phrase(
            response,
            phrase,
        ):

            failures.append(
                f"Missing required content: '{phrase}'"
            )

    # must_include_concepts:
    # Meaning/behavior-level checks.
    for concept in expectations.get(
        "must_include_concepts",
        [],
    ):

        if not concept_satisfied(
            response,
            concept,
        ):

            failures.append(
                f"Missing required concept: '{concept}'"
            )

    return (
        len(failures) == 0,
        failures,
    )


# ---------------------------------------------------------
# Forbidden content
# ---------------------------------------------------------

def check_forbidden_content(
    response: str,
    expectations: dict,
) -> tuple[bool, list[str]]:

    failures = []

    for phrase in expectations.get(
        "must_not_include",
        [],
    ):

        if contains_phrase(
            response,
            phrase,
        ):

            failures.append(
                f"Forbidden content found: '{phrase}'"
            )

    for phrase in expectations.get(
        "must_not_invent",
        [],
    ):

        if contains_phrase(
            response,
            phrase,
        ):

            failures.append(
                f"Forbidden invented content found: '{phrase}'"
            )

    return (
        len(failures) == 0,
        failures,
    )


# ---------------------------------------------------------
# Source checking
# ---------------------------------------------------------

def normalize_source_name(
    source: str,
) -> str:

    source = normalize_text(source)

    source = source.replace(
        "document:",
        "",
    )

    source = source.replace(
        "source:",
        "",
    )

    return source.strip()


def check_sources(
    response: str,
    expectations: dict,
) -> tuple[bool, list[str]]:

    failures = []

    response_n = normalize_text(
        response
    )

    required_sources = expectations.get(
        "required_sources",
        [],
    )

    forbidden_sources = expectations.get(
        "forbidden_sources_as_authority",
        [],
    )

    # Required sources

    for source in required_sources:

        source_n = normalize_source_name(
            source
        )

        if source_n not in response_n:

            filename = Path(
                source_n
            ).name

            if filename not in response_n:

                failures.append(
                    f"Required source missing: '{source}'"
                )

    # Forbidden sources

    for source in forbidden_sources:

        source_n = normalize_source_name(
            source
        )

        if source_n in response_n:

            failures.append(
                f"Forbidden source referenced: '{source}'"
            )

    return (
        len(failures) == 0,
        failures,
    )


# ---------------------------------------------------------
# Handoff checking
# ---------------------------------------------------------

def check_handoff(
    response: str,
    expectations: dict,
) -> tuple[bool, list[str]]:

    expected = expectations.get(
        "handoff"
    )

    if expected is None:
        return True, []

    text = normalize_text(
        response
    )

    handoff_indicators = [
        "contact a support specialist",
        "contact support",
        "support specialist",
        "human assistance",
        "human review",
        "contact customer support",
        "speak with a support",
        "recommend contacting",
        "requires human",
        "escalate to support",
        "escalate this",
        "manual review",
    ]

    detected = any(
        phrase in text
        for phrase in handoff_indicators
    )

    if expected and not detected:

        return False, [
            "Expected human handoff/recommendation "
            "but none was detected."
        ]

    if not expected and detected:

        return False, [
            "Unexpected human handoff/recommendation "
            "was detected."
        ]

    return True, []


# ---------------------------------------------------------
# Agent debug output
# ---------------------------------------------------------

def extract_agent_action(
    debug_output: str,
) -> str | None:

    for line in debug_output.splitlines():

        stripped = line.strip()

        if stripped.lower().startswith(
            "action:"
        ):

            return stripped.split(
                ":",
                1,
            )[1].strip().lower()

    return None


def extract_order_id(
    debug_output: str,
) -> str | None:

    for line in debug_output.splitlines():

        stripped = line.strip()

        if stripped.lower().startswith(
            "order id:"
        ):

            value = stripped.split(
                ":",
                1,
            )[1].strip()

            if value.lower() == "none":
                return None

            return value

    return None


# ---------------------------------------------------------
# Tool behavior
# ---------------------------------------------------------

def check_tool_behavior(
    expectations: dict,
    action: str | None,
    debug_output: str,
) -> tuple[bool, list[str]]:

    expected_tool = expectations.get(
        "tool"
    )

    failures = []

    if not expected_tool:
        return True, failures

    # No tool should be called

    if expected_tool == "not_called":

        if action in {
            "order",
            "order_and_knowledge",
        }:

            failures.append(
                "Expected no order lookup, "
                f"but agent action was '{action}'."
            )

        return (
            len(failures) == 0,
            failures,
        )

    # Order lookup

    if expected_tool == "order_lookup":

        if action not in {
            "order",
            "order_and_knowledge",
        }:

            failures.append(
                "Expected order lookup, "
                f"but agent action was '{action}'."
            )

        expected_args = expectations.get(
            "tool_arguments",
            {},
        )

        expected_order_id = expected_args.get(
            "order_id"
        )

        if expected_order_id:

            actual_order_id = extract_order_id(
                debug_output
            )

            if actual_order_id is None:

                failures.append(
                    f"Expected order ID '{expected_order_id}' "
                    "to be passed to lookup, "
                    "but no order ID was detected."
                )

            elif (
                actual_order_id.upper()
                != expected_order_id.upper()
            ):

                failures.append(
                    "Wrong order ID passed to lookup: "
                    f"expected {expected_order_id}, "
                    f"got {actual_order_id}"
                )

        return (
            len(failures) == 0,
            failures,
        )

    return True, failures


# ---------------------------------------------------------
# Rate-aware agent invocation
# ---------------------------------------------------------

def run_agent_rate_aware(
    content: str,
    history: list[dict],
):

    global _last_agent_call

    for attempt in range(
        MAX_429_RETRIES + 1
    ):

        elapsed = (
            time.monotonic()
            - _last_agent_call
        )

        wait_for = (
            MIN_SECONDS_BETWEEN_AGENT_CALLS
            - elapsed
        )

        if wait_for > 0:
            time.sleep(wait_for)

        try:

            debug_buffer = io.StringIO()

            with redirect_stdout(
                debug_buffer
            ):

                response = run_agent(
                    content,
                    history,
                )

            _last_agent_call = (
                time.monotonic()
            )

            return (
                response,
                debug_buffer.getvalue(),
            )

        except Exception as error:

            _last_agent_call = (
                time.monotonic()
            )

            error_text = str(
                error
            ).lower()

            is_rate_limit = (
                "429" in error_text
                or "resource_exhausted"
                in error_text
                or "quota" in error_text
                or "rate limit"
                in error_text
            )

            if (
                not is_rate_limit
                or attempt >= MAX_429_RETRIES
            ):

                raise

            backoff = 15 * (
                attempt + 1
            )

            print(
                f"  Rate limit hit. "
                f"Waiting {backoff}s before retry "
                f"{attempt + 1}/{MAX_429_RETRIES}..."
            )

            time.sleep(
                backoff
            )

    raise RuntimeError(
        "Unreachable"
    )


# ---------------------------------------------------------
# Single case execution
# ---------------------------------------------------------

def run_case(
    case: dict,
) -> dict:

    case_id = case.get(
        "id",
        "unknown-case",
    )

    category = case.get(
        "category",
        "uncategorized",
    )

    messages = case.get(
        "messages",
        [],
    )

    expectations = case.get(
        "expect",
        {},
    )

    history = []
    responses = []
    debug_outputs = []
    case_failures = []

    if not isinstance(
        messages,
        list,
    ):

        return {
            "id": case_id,
            "category": category,
            "passed": False,
            "response": "",
            "action": None,
            "failures": [
                "Case 'messages' must be a list."
            ],
        }

    # Same conversation for every message.

    for message in messages:

        if message.get(
            "role"
        ) != "user":

            continue

        content = message.get(
            "content",
            "",
        )

        try:

            response, debug_output = (
                run_agent_rate_aware(
                    content,
                    history,
                )
            )

            responses.append(
                response
            )

            debug_outputs.append(
                debug_output
            )

            history.append({
                "role": "user",
                "content": content,
            })

            history.append({
                "role": "assistant",
                "content": response,
            })

        except Exception as error:

            return {
                "id": case_id,
                "category": category,
                "passed": False,
                "response": "",
                "action": None,
                "failures": [
                    "Agent error: "
                    f"{type(error).__name__}: "
                    f"{error}"
                ],
            }

    final_response = "\n".join(
        responses
    )

    combined_debug = "\n".join(
        debug_outputs
    )

    # -----------------------------------------------------
    # Deterministic assertions
    # -----------------------------------------------------

    _, failures = check_required_content(
        final_response,
        expectations,
    )

    case_failures.extend(
        failures
    )

    _, failures = check_forbidden_content(
        final_response,
        expectations,
    )

    case_failures.extend(
        failures
    )

    _, failures = check_sources(
        final_response,
        expectations,
    )

    case_failures.extend(
        failures
    )

    _, failures = check_handoff(
        final_response,
        expectations,
    )

    case_failures.extend(
        failures
    )

    # -----------------------------------------------------
    # Tool behavior
    # -----------------------------------------------------

    action = None

    # Last detected action wins for multi-turn cases.

    for debug_output in debug_outputs:

        detected = extract_agent_action(
            debug_output
        )

        if detected:
            action = detected

    _, failures = check_tool_behavior(
        expectations,
        action,
        combined_debug,
    )

    case_failures.extend(
        failures
    )

    return {
        "id": case_id,
        "category": category,
        "passed": len(case_failures) == 0,
        "response": final_response,
        "action": action,
        "failures": case_failures,
    }


# ---------------------------------------------------------
# Category summary
# ---------------------------------------------------------

def print_category_summary(
    results: list[dict],
):

    categories = {}

    for result in results:

        category = result[
            "category"
        ]

        if category not in categories:

            categories[category] = {
                "passed": 0,
                "failed": 0,
            }

        if result[
            "passed"
        ]:

            categories[
                category
            ]["passed"] += 1

        else:

            categories[
                category
            ]["failed"] += 1

    print()
    print("=" * 70)
    print("CATEGORY RESULTS")
    print("=" * 70)

    for category, stats in sorted(
        categories.items()
    ):

        total = (
            stats["passed"]
            + stats["failed"]
        )

        percentage = (
            stats["passed"]
            / total
            * 100
            if total
            else 0
        )

        print(
            f"{category:25} "
            f"{stats['passed']}/{total} "
            f"({percentage:.1f}%)"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("AI SUPPORT AGENT EVALUATION")
    print("=" * 70)

    print(
        "Rate protection: one agent call every "
        f"{MIN_SECONDS_BETWEEN_AGENT_CALLS:.1f}s"
    )

    # -----------------------------------------------------
    # Load cases
    # -----------------------------------------------------

    visible_cases = load_cases(
        VISIBLE_CASES_FILE
    )

    custom_cases = load_cases(
        CUSTOM_CASES_FILE
    )

    all_cases = (
        visible_cases
        + custom_cases
    )

    print()

    print(
        f"Visible cases : "
        f"{len(visible_cases)}"
    )

    print(
        f"Custom cases  : "
        f"{len(custom_cases)}"
    )

    print(
        f"Total cases   : "
        f"{len(all_cases)}"
    )

    if not all_cases:

        print()
        print(
            "ERROR: No evaluation cases found."
        )

        return 1

    # -----------------------------------------------------
    # Run cases
    # -----------------------------------------------------

    results = []

    for index, case in enumerate(
        all_cases,
        start=1,
    ):

        print()
        print("=" * 70)

        print(
            f"CASE {index}/{len(all_cases)}: "
            f"{case.get('id', 'unknown')}"
        )

        print("=" * 70)

        result = run_case(
            case
        )

        results.append(
            result
        )

        if result[
            "passed"
        ]:

            print(
                "PASS"
            )

        else:

            print(
                "FAIL"
            )

            for failure in result[
                "failures"
            ]:

                print(
                    f"  - {failure}"
                )

        if result[
            "response"
        ]:

            print()
            print(
                "ANSWER"
            )

            print(
                "-" * 70
            )

            print(
                result[
                    "response"
                ]
            )

    # -----------------------------------------------------
    # Overall summary
    # -----------------------------------------------------

    passed = sum(
        1
        for result in results
        if result[
            "passed"
        ]
    )

    failed = (
        len(results)
        - passed
    )

    percentage = (
        passed
        / len(results)
        * 100
        if results
        else 0
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Passed : {passed}"
    )

    print(
        f"Failed : {failed}"
    )

    print(
        f"Total  : {len(results)}"
    )

    print(
        f"Score  : {percentage:.1f}%"
    )

    print_category_summary(
        results
    )

    # -----------------------------------------------------
    # Failed cases
    # -----------------------------------------------------

    failed_results = [
        result
        for result in results
        if not result[
            "passed"
        ]
    ]

    if failed_results:

        print()
        print("=" * 70)
        print("FAILED CASES")
        print("=" * 70)

        for result in failed_results:

            print()

            print(
                f"{result['id']} "
                f"[{result['category']}]"
            )

            for failure in result[
                "failures"
            ]:

                print(
                    f"  - {failure}"
                )

    print()
    print("=" * 70)

    return (
        0
        if failed == 0
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
