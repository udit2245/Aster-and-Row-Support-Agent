from datetime import datetime, timezone
import re


STATUS_PRIORITY = {
    "active": 4,
    "current": 4,
    "draft": 1,
    "superseded": 0,
    "expired": 0,
}

AUTHORITY_PRIORITY = {
    "official": 3,
    "internal": 1,
    "third-party": 1,
    "none": 0,
}


def _parse_date(value: str | None):
    if not value:
        return None

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d"
        ).date()
    except ValueError:
        return None


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def _query_topics(query: str) -> set[str]:
    """
    Extract broad business topics from the query.
    """

    topic_map = {
        "return": {
            "return",
            "returns",
            "refund",
            "refunds",
            "eligible",
        },
        "cancel": {
            "cancel",
            "cancellation",
            "cancelled",
        },
        "shipping": {
            "ship",
            "shipping",
            "delivery",
            "deliver",
        },
        "damaged": {
            "damaged",
            "defective",
            "wrong",
        },
        "warranty": {
            "warranty",
            "defect",
        },
        "membership": {
            "trailplus",
            "membership",
        },
        "gift": {
            "gift",
            "giftcard",
            "giftcards",
        },
        "price": {
            "price",
            "adjustment",
            "discount",
        },
        "address": {
            "address",
            "change",
        },
    }

    query_tokens = _tokenize(query)

    topics = set()

    for topic, words in topic_map.items():

        if query_tokens.intersection(words):
            topics.add(topic)

    return topics


def _document_topics(document: dict) -> set[str]:

    metadata = document.get("metadata", {})

    searchable = " ".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("section", "")),
            str(document.get("text", "")),
        ]
    )

    return _query_topics(searchable)


def build_policy_query(
    question: str,
    order: dict | None = None
) -> str:
    """
    Expand the user's question with structured order context.

    This improves retrieval for questions such as:
    'Can I return ORD-1001?'
    where order status/date/membership can determine which
    policy is relevant.
    """

    parts = [question]

    if order:

        status = order.get("status")
        membership = order.get("membership_tier")
        placed_at = order.get("placed_at")

        if status:
            parts.append(f"order status {status}")

        if membership:
            parts.append(
                f"membership tier {membership}"
            )

        if placed_at:
            parts.append(
                f"order placed {placed_at}"
            )

    return " ".join(parts)


def _is_customer_usable(metadata: dict) -> bool:

    status = str(
        metadata.get("status", "")
    ).lower()

    authority = str(
        metadata.get("policy_authority", "")
    ).lower()

    customer_answering = metadata.get(
        "customer_answering"
    )

    # Explicitly non-customer content.
    if customer_answering is not None:
        if str(customer_answering).lower() == "false":
            return False

    # Draft/internal scratch content should not be
    # used as customer policy.
    if status == "draft":
        return False

    if authority == "none":
        return False

    return True


def _status_applicable(
    metadata: dict,
    order: dict | None
) -> bool:

    status = str(
        metadata.get("status", "")
    ).lower()

    if status in {"active", "current"}:
        return True

    if status not in {"superseded", "expired"}:
        return False

    # Historical policies are allowed when an order
    # was placed while that policy was active.
    if not order:
        return False

    placed_at = order.get("placed_at")

    order_date = (
        datetime.fromisoformat(
            placed_at.replace("Z", "+00:00")
        ).date()
        if placed_at
        else None
    )

    effective_date = _parse_date(
        metadata.get("effective_date")
    )

    superseded_date = _parse_date(
        metadata.get("superseded_date")
    )

    if not order_date:
        return False

    if effective_date and order_date < effective_date:
        return False

    if superseded_date and order_date >= superseded_date:
        return False

    return True


def _score_document(
    document: dict,
    query: str,
    order: dict | None
) -> float:

    metadata = document.get("metadata", {})

    status = str(
        metadata.get("status", "")
    ).lower()

    authority = str(
        metadata.get("policy_authority", "")
    ).lower()

    query_topics = _query_topics(query)
    document_topics = _document_topics(document)

    score = 0.0

    # ---------------------------------------------
    # Retrieval similarity
    # ---------------------------------------------

    distance = document.get(
        "distance",
        1.0
    )

    similarity = 1 / (1 + distance)

    score += similarity * 5

    # ---------------------------------------------
    # Policy status
    # ---------------------------------------------

    score += STATUS_PRIORITY.get(
        status,
        0
    ) * 4

    # ---------------------------------------------
    # Authority
    # ---------------------------------------------

    score += AUTHORITY_PRIORITY.get(
        authority,
        0
    ) * 2

    # ---------------------------------------------
    # Topic overlap
    # ---------------------------------------------

    overlap = query_topics.intersection(
        document_topics
    )

    score += len(overlap) * 6

    # ---------------------------------------------
    # Order-aware relevance
    # ---------------------------------------------

    if order:

        order_status = str(
            order.get("status", "")
        ).lower()

        title = str(
            metadata.get("title", "")
        ).lower()

        section = str(
            metadata.get("section", "")
        ).lower()

        combined = f"{title} {section}"

        if order_status == "pending":

            if (
                "cancellation" in combined
                or "cancel" in combined
            ):
                score += 15

        if (
            order.get("membership_tier", "").lower()
            == "standard"
        ):

            if "trailplus" in combined:
                score -= 4

    return score


def select_policies(
    documents: list[dict],
    query: str,
    order: dict | None = None,
    max_documents: int = 5
) -> list[dict]:
    """
    Filter and rank retrieved policy documents.
    """

    candidates = []

    for document in documents:

        metadata = document.get(
            "metadata",
            {}
        )

        if not _is_customer_usable(metadata):
            continue

        if not _status_applicable(
            metadata,
            order
        ):
            continue

        score = _score_document(
            document,
            query,
            order
        )

        candidate = document.copy()

        candidate["policy_score"] = score

        candidates.append(candidate)

    candidates.sort(
        key=lambda item: item["policy_score"],
        reverse=True
    )

    return candidates[:max_documents]