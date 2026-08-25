from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Document:
    content: str
    source: str
    metadata: dict


def parse_front_matter(text: str):
    """
    Separate YAML front matter from the Markdown content.
    """

    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)

    if len(parts) != 3:
        return {}, text

    front_matter = yaml.safe_load(parts[1]) or {}
    content = parts[2].strip()

    return front_matter, content


def load_documents(directory: str) -> list[Document]:
    documents = []

    directory_path = Path(directory)

    for file_path in directory_path.rglob("*.md"):
        raw_text = file_path.read_text(encoding="utf-8")

        metadata, content = parse_front_matter(raw_text)

        documents.append(
            Document(
                content=content,
                source=file_path.name,
                metadata=metadata
            )
        )

    return documents