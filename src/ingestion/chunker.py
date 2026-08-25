from dataclasses import dataclass
from src.document_loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict


def chunk_document(document: Document, max_chars: int = 1200) -> list[Chunk]:
    """
    Split a document into heading-aware chunks.
    """

    lines = document.content.splitlines()

    chunks = []
    current_heading = ""
    current_lines = []

    for line in lines:

        if line.startswith("#"):
            if current_lines:
                text = "\n".join(current_lines).strip()

                if text:
                    metadata = document.metadata.copy()
                    metadata["section"] = current_heading

                    chunks.append(
                        Chunk(
                            text=text,
                            source=document.source,
                            metadata=metadata
                        )
                    )

            current_heading = line.lstrip("#").strip()
            current_lines = [line]

        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()

        if text:
            metadata = document.metadata.copy()
            metadata["section"] = current_heading

            chunks.append(
                Chunk(
                    text=text,
                    source=document.source,
                    metadata=metadata
                )
            )

    # Break oversized sections
    final_chunks = []

    for chunk in chunks:
        if len(chunk.text) <= max_chars:
            final_chunks.append(chunk)
            continue

        text = chunk.text

        for i in range(0, len(text), max_chars):
            piece = text[i:i + max_chars]

            final_chunks.append(
                Chunk(
                    text=piece,
                    source=chunk.source,
                    metadata=chunk.metadata.copy()
                )
            )

    return final_chunks