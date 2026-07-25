# rag_backend.py
# PDF RAG backend — uses only pypdf + numpy (no torch/sentence-transformers needed).
# Retrieval is TF-IDF cosine similarity: fast, lightweight, accurate for keyword/concept matching.

import re
import math
from collections import Counter

import numpy as np
from pypdf import PdfReader
import io

# ─────────────────────────────────────────────────────────────
# Internal state (one PDF at a time, global)
# ─────────────────────────────────────────────────────────────
_chunks: list[str] = []          # raw text chunks
_chunk_meta: list[dict] = []     # page numbers etc.
_tfidf_matrix: np.ndarray | None = None   # shape (n_chunks, vocab)
_vocab: dict[str, int] = {}      # term → column index
_idf: np.ndarray | None = None
_loaded_filename: str | None = None


# ─────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_text(pages: list[tuple[int, str]], chunk_size: int = 500, overlap: int = 50):
    """Split page text into overlapping chunks, keeping page number metadata."""
    chunks, meta = [], []
    for page_num, text in pages:
        words = text.split()
        i = 0
        while i < len(words):
            window = words[i : i + chunk_size]
            chunks.append(" ".join(window))
            meta.append({"page": page_num})
            i += chunk_size - overlap
    return chunks, meta


# ─────────────────────────────────────────────────────────────
# TF-IDF index
# ─────────────────────────────────────────────────────────────
def _build_index(chunks: list[str]):
    """Build TF-IDF matrix from chunks."""
    tokenized = [_tokenize(c) for c in chunks]

    # Build vocabulary (terms that appear in ≥2 chunks, to filter noise)
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    vocab = {term: idx for idx, (term, count) in enumerate(df.items()) if count >= 1}

    N = len(chunks)
    V = len(vocab)

    # TF matrix
    tf_matrix = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        total = len(tokens) or 1
        for term, cnt in counts.items():
            if term in vocab:
                tf_matrix[i, vocab[term]] = cnt / total

    # IDF vector
    doc_freq = np.array([df.get(term, 0) for term in vocab], dtype=np.float32)
    idf = np.log((N + 1) / (doc_freq + 1)) + 1.0  # smoothed IDF

    # TF-IDF
    tfidf = tf_matrix * idf

    # L2-normalise rows
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf = tfidf / norms

    return vocab, idf, tfidf


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def process_pdf(file_bytes: bytes, filename: str) -> int:
    """
    Load a PDF from raw bytes, chunk it, and build a TF-IDF index.
    Returns the number of chunks indexed.
    """
    global _chunks, _chunk_meta, _tfidf_matrix, _vocab, _idf, _loaded_filename

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_num + 1, text))

    _chunks, _chunk_meta = _chunk_text(pages, chunk_size=150, overlap=20)
    _vocab, _idf, _tfidf_matrix = _build_index(_chunks)
    _loaded_filename = filename

    return len(_chunks)


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Return the top-k most relevant chunks from the loaded PDF using TF-IDF cosine similarity.
    """
    if _tfidf_matrix is None or not _chunks:
        return "No PDF has been uploaded yet. Ask the user to upload a PDF first."

    # Build query vector
    q_tokens = _tokenize(query)
    q_counts = Counter(q_tokens)
    q_vec = np.zeros(len(_vocab), dtype=np.float32)
    for term, cnt in q_counts.items():
        if term in _vocab:
            q_vec[_vocab[term]] = cnt
    q_vec = q_vec * _idf
    norm = np.linalg.norm(q_vec)
    if norm == 0:
        return "Could not find relevant content for that query."
    q_vec /= norm

    # Cosine similarity (dot product since rows are L2-normalised)
    scores = _tfidf_matrix @ q_vec
    top_indices = np.argsort(scores)[::-1][:k]

    passages = []
    for idx in top_indices:
        page = _chunk_meta[idx].get("page", "?")
        passages.append(f"[Page {page}]\n{_chunks[idx]}")

    return "\n\n---\n\n".join(passages)


def get_loaded_filename() -> str | None:
    return _loaded_filename
