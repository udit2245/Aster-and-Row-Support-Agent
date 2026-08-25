import chromadb

from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import embed_documents
from src.config import CHROMA_PATH


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


def get_collection():
    return client.get_or_create_collection(
        name="aster_row_knowledge"
    )


def reset_collection():
    """
    Delete and recreate the knowledge-base collection.
    Used when rebuilding the index.
    """

    try:
        client.delete_collection(
            name="aster_row_knowledge"
        )
        print("Existing collection deleted.")
    except Exception:
        pass

    return get_collection()


def index_chunks(chunks: list[Chunk]):
    """
    Embed and store chunks in ChromaDB.
    """

    collection = reset_collection()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = embed_documents(texts)

    ids = [
        f"chunk-{i}"
        for i in range(len(chunks))
    ]

    metadatas = []

    for chunk in chunks:

        metadata = chunk.metadata.copy()

        metadata["source"] = chunk.source

        metadata = {
            key: str(value)
            for key, value in metadata.items()
            if value is not None
        }

        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Indexed {len(chunks)} chunks successfully."
    )