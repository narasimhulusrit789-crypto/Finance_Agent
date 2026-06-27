"""
RAG (Retrieval-Augmented Generation) Pipeline
────────────────────────────────────────────────────────────────
Handles:
  • Document ingestion (PDF, DOCX, TXT, CSV, XLSX)
  • Text chunking and embedding
  • Vector store management (ChromaDB / FAISS)
  • Semantic similarity search
  • Context assembly for Granite prompts
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import structlog
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  Text splitter (no external dependency)
# ─────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


# ─────────────────────────────────────────────────────────────
#  Document loaders
# ─────────────────────────────────────────────────────────────

def _load_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: str | Path) -> str:
    try:
        import PyPDF2
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except ImportError:
        logger.warning("PyPDF2 not installed — cannot parse PDF")
        return ""


def _load_docx(path: str | Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs)
    except ImportError:
        logger.warning("python-docx not installed — cannot parse DOCX")
        return ""


def _load_csv(path: str | Path) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("csv.load_failed", error=str(exc))
        return ""


def _load_xlsx(path: str | Path) -> str:
    try:
        import pandas as pd
        dfs = pd.read_excel(path, sheet_name=None)
        parts = []
        for sheet, df in dfs.items():
            parts.append(f"[Sheet: {sheet}]\n{df.to_string(index=False)}")
        return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("xlsx.load_failed", error=str(exc))
        return ""


_LOADERS: dict[str, Any] = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".csv": _load_csv,
    ".xlsx": _load_xlsx,
    ".xls": _load_xlsx,
}


def load_document(path: str | Path) -> str:
    """Load a document from disk and return its text content."""
    p = Path(path)
    loader = _LOADERS.get(p.suffix.lower())
    if not loader:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return loader(p)


# ─────────────────────────────────────────────────────────────
#  Vector store abstraction
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around ChromaDB for financial document storage.
    Falls back to an in-memory FAISS-style store when ChromaDB
    is unavailable.
    """

    def __init__(self) -> None:
        cfg = get_settings().vector_db
        self._cfg = cfg
        self._embedder = SentenceTransformer(cfg.embedding_model)
        self._collection = None
        self._init_store()
        logger.info("VectorStore.init", type=cfg.db_type, path=cfg.db_path)

    def _init_store(self) -> None:
        try:
            import chromadb
            os.makedirs(self._cfg.db_path, exist_ok=True)
            client = chromadb.PersistentClient(path=self._cfg.db_path)
            self._collection = client.get_or_create_collection(
                name=self._cfg.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chromadb"
        except Exception as exc:  # noqa: BLE001
            logger.warning("chromadb.init_failed", error=str(exc), fallback="in-memory")
            self._collection = None
            self._backend = "memory"
            self._memory_store: list[dict] = []

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        source: str = "unknown",
    ) -> int:
        """Embed and store a list of text chunks. Returns count stored."""
        if not texts:
            return 0
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()
        metas = metadatas or [{"source": source}] * len(texts)
        ids = [hashlib.md5(f"{source}_{i}_{t[:50]}".encode()).hexdigest() for i, t in enumerate(texts)]

        if self._backend == "chromadb":
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metas,
            )
        else:
            for i, (text, emb, meta, doc_id) in enumerate(zip(texts, embeddings, metas, ids)):
                self._memory_store.append({"id": doc_id, "text": text, "embedding": emb, "metadata": meta})

        logger.debug("vector_store.added", n=len(texts), source=source)
        return len(texts)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Perform semantic similarity search and return top-k results."""
        q_emb = self._embedder.encode([query_text], show_progress_bar=False).tolist()[0]

        if self._backend == "chromadb":
            results = self._collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k, max(self._collection.count(), 1)),
                include=["documents", "metadatas", "distances"],
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            return [
                {
                    "text": doc,
                    "source": m.get("source", "unknown"),
                    "score": 1 - dist,  # cosine similarity
                }
                for doc, m, dist in zip(docs, metas, distances)
            ]
        else:
            # Brute-force cosine similarity in memory
            import numpy as np
            if not self._memory_store:
                return []
            q = np.array(q_emb)
            scored = []
            for item in self._memory_store:
                v = np.array(item["embedding"])
                sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
                scored.append({**item, "score": sim})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return [
                {"text": s["text"], "source": s["metadata"].get("source", "unknown"), "score": s["score"]}
                for s in scored[:top_k]
            ]

    def count(self) -> int:
        if self._backend == "chromadb":
            return self._collection.count()
        return len(self._memory_store)

    def reset(self) -> None:
        """Clear all documents from the store."""
        if self._backend == "chromadb":
            self._collection.delete(where={"source": {"$ne": "__never__"}})
        else:
            self._memory_store.clear()
        logger.info("vector_store.reset")


# ─────────────────────────────────────────────────────────────
#  RAG Retriever (used by InvestmentKnowledgeAgent)
# ─────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    High-level retriever used by agents.
    Wraps VectorStore with async query support and context assembly.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store or VectorStore()

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant document chunks asynchronously."""
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._store.query, query, top_k)
        return [r for r in results if r.get("score", 0) > 0.3]

    def assemble_context(self, results: list[dict], max_chars: int = 3000) -> str:
        """Combine retrieved chunks into a single context block."""
        parts = []
        total = 0
        for r in results:
            text = r["text"]
            if total + len(text) > max_chars:
                text = text[: max_chars - total]
            parts.append(f"[Source: {r.get('source', 'KB')}]\n{text}")
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
#  RAG Ingestion Pipeline
# ─────────────────────────────────────────────────────────────

class RAGIngestionPipeline:
    """
    Ingests financial documents into the vector store.
    Supports single-file or batch directory ingestion.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        cfg = get_settings().vector_db
        self._store = vector_store or VectorStore()
        self._chunk_size = cfg.chunk_size
        self._chunk_overlap = cfg.chunk_overlap

    def ingest_file(self, path: str | Path, source_label: str | None = None) -> int:
        """Load, chunk, embed, and store a single document. Returns chunk count."""
        p = Path(path)
        source = source_label or p.name
        logger.info("rag.ingest", file=str(p), source=source)
        text = load_document(p)
        if not text.strip():
            logger.warning("rag.ingest.empty", file=str(p))
            return 0
        chunks = _split_text(text, self._chunk_size, self._chunk_overlap)
        metadatas = [{"source": source, "file": str(p), "chunk": i} for i in range(len(chunks))]
        return self._store.add_documents(chunks, metadatas, source=source)

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> dict[str, int]:
        """Ingest all supported documents in a directory."""
        d = Path(directory)
        pattern = "**/*" if recursive else "*"
        results: dict[str, int] = {}
        for fp in d.glob(pattern):
            if fp.is_file() and fp.suffix.lower() in _LOADERS:
                try:
                    n = self.ingest_file(fp)
                    results[fp.name] = n
                except Exception as exc:  # noqa: BLE001
                    logger.error("rag.ingest.error", file=str(fp), error=str(exc))
                    results[fp.name] = -1
        return results

    def ingest_bytes(self, data: bytes, filename: str) -> int:
        """Ingest an in-memory file upload."""
        suffix = Path(filename).suffix.lower()
        tmp_path = Path(f"/tmp/{filename}")
        tmp_path.write_bytes(data)
        try:
            return self.ingest_file(tmp_path, source_label=filename)
        finally:
            tmp_path.unlink(missing_ok=True)

    def ingest_text(self, text: str, source: str = "manual") -> int:
        """Directly ingest raw text (e.g., scraped news or user-provided notes)."""
        chunks = _split_text(text, self._chunk_size, self._chunk_overlap)
        metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
        return self._store.add_documents(chunks, metadatas, source=source)

    @property
    def document_count(self) -> int:
        return self._store.count()
