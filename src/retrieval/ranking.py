STATUS_PRIORITY = {
    "active": 3,
    "current": 3,
    "draft": 2,
    "superseded": 1,
    "expired": 0,
}


AUTHORITY_PRIORITY = {
    "official": 3,
    "internal": 2,
    "third-party": 1,
}


def rank_documents(documents: list[dict]) -> list[dict]:
    """
    Rank retrieved documents using policy authority and status.

    Higher priority documents should be preferred over
    outdated or superseded documents.
    """

    def score(document):

        metadata = document["metadata"]

        status = metadata.get("status", "").lower()
        authority = metadata.get("policy_authority", "").lower()

        status_score = STATUS_PRIORITY.get(status, 0)
        authority_score = AUTHORITY_PRIORITY.get(authority, 0)

        # Retrieval similarity still matters.
        # Chroma distance: lower is better.
        distance = document.get("distance", 1.0)

        similarity_score = 1 / (1 + distance)

        return (
            status_score,
            authority_score,
            similarity_score
        )

    return sorted(
        documents,
        key=score,
        reverse=True
    )