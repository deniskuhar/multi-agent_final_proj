from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import get_settings


@dataclass
class SearchResult:
    source: str
    page: int | None
    text: str


class MarketResearchRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = self.settings.index_path
        self.faiss_dir = self.index_dir / "faiss_index"
        self.chunks_path = self.index_dir / "chunks.pkl"

        if not self.faiss_dir.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.faiss_dir}. Run `python ingest.py` first."
            )
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found at {self.chunks_path}. Run `python ingest.py` first."
            )

        embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key.get_secret_value(),
        )

        self.vector_store = FAISS.load_local(
            str(self.faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        with self.chunks_path.open("rb") as f:
            self.chunks: list[Document] = pickle.load(f)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search(query, k=top_k)

    def search_with_scores(self, query: str, k: int | None = None):
        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search_with_score(query, k=top_k)

    def formatted_search(self, query: str, k: int | None = None, max_chars: int = 700) -> str:
        docs = self.search(query, k=k)
        if not docs:
            return f"No local RAG results found for query: {query}"

        lines = [f"Found {len(docs)} local results for query: {query}"]
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            text = doc.page_content.strip().replace("\n", " ")
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            lines.append(f"{i}. [{source}{page_label}] {text}")

        return "\n".join(lines)


_RETRIEVER: MarketResearchRetriever | None = None


def get_retriever() -> MarketResearchRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = MarketResearchRetriever()
    return _RETRIEVER