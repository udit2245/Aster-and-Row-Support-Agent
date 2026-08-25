from src.document_loader import load_documents
from src.ingestion.chunker import chunk_document
from src.retrieval.vector_store import index_chunks


def main():

    print("=" * 70)
    print("ASTER & ROW KNOWLEDGE BASE INDEXER")
    print("=" * 70)

    print("\nLoading documents...")

    documents = load_documents(
        "data/knowledge-base"
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    print("\nCreating chunks...")

    chunks = []

    for document in documents:

        document_chunks = chunk_document(
            document
        )

        chunks.extend(
            document_chunks
        )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("\nIndexing knowledge base...")

    index_chunks(
        chunks
    )

    print("\nKnowledge base ready.")


if __name__ == "__main__":
    main()