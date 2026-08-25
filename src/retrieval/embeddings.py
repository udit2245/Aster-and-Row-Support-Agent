from google import genai
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)


def embed_text(text: str) -> list[float]:

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


def embed_documents(texts: list[str]) -> list[list[float]]:

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]