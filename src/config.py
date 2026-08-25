import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

# Project paths
KNOWLEDGE_BASE_PATH = "data/knowledge-base"
CHROMA_PATH = ".chroma"

# Models
LLM_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"