# retriever.py
"""
Step 2: Connects to the saved ChromaDB vector store and returns a LangChain vector retriever
that uses the same multilingual embedding model as ingest.py.
"""
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_store")
COLLECTION_NAME = "bangla_book_rag"
EMBED_MODEL = "BAAI/bge-m3"  # must match ingest.py


def get_retriever(k: int = 10):
    """Returns a LangChain VectorStoreRetriever that fetches the top-k most relevant chunks."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DIR}'. Please run 'python ingest.py' first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # must match ingest.py
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if vectorstore._collection.count() == 0:
        raise ValueError("The vector store is empty. Please run 'python ingest.py' first.")

    # Standard LangChain similarity retriever
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )