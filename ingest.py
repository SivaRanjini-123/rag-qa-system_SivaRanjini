import os
from app.vector_store import save_embeddings

def load_document(file_path: str):
    print(f"[INGEST] Loading file: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunk_size = 500
    overlap = 50

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    print(f"[INGEST] Chunks created: {len(chunks)}")

    save_embeddings(chunks)
