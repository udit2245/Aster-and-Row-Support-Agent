from src.retrieval.embeddings import embed_text
from src.retrieval.vector_store import get_collection


def retrieve(
    query: str,
    top_k: int = 8
):
    """
    Retrieve relevant knowledge-base chunks.
    """

    collection = get_collection()

    query_embedding = embed_text(
        query
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    for i in range(len(documents)):

        retrieved.append(
            {
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            }
        )

    return retrieved