from __future__ import annotations

import json
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import get_settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_text_file(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "path": str(path),
                "page": None,
            },
        )
    ]


def load_pdf_file(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["path"] = str(path)
    return docs


def load_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            loaded = load_pdf_file(path)
        else:
            loaded = load_text_file(path)

        documents.extend(loaded)

    return documents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - chunk_overlap)

    return chunks


def prepare_chunks(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunks: list[Document] = []

    for doc in documents:
        text_chunks = chunk_text(doc.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(text_chunks):
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = idx
            chunks.append(Document(page_content=chunk, metadata=metadata))

    return chunks


def ingest() -> None:
    settings = get_settings()

    data_dir = settings.data_path
    index_dir = settings.index_path
    faiss_dir = index_dir / "faiss_index"
    chunks_path = index_dir / "chunks.pkl"
    manifest_path = index_dir / "ingest_manifest.json"

    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    settings.output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from: {data_dir}")
    documents = load_documents(data_dir)
    if not documents:
        raise FileNotFoundError(f"No supported documents found in {data_dir}")

    print(f"Loaded {len(documents)} document pages/files")

    chunks = prepare_chunks(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No chunks were produced during ingestion")

    print(f"Prepared {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )

    print(f"Building FAISS index with embeddings model: {settings.embedding_model}")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(faiss_dir))

    with chunks_path.open("wb") as f:
        pickle.dump(chunks, f)

    manifest = {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "embedding_model": settings.embedding_model,
        "data_dir": str(data_dir),
        "faiss_dir": str(faiss_dir),
        "chunks_path": str(chunks_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved FAISS index to: {faiss_dir}")
    print(f"Saved chunks to: {chunks_path}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    ingest()