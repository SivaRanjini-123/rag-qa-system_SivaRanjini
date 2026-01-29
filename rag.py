import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "data/faiss_index/index.faiss"
CHUNKS_PATH = "data/faiss_index/chunks.pkl"

def answer_question(query: str, k: int = 3):
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        return "No documents found. Please upload files first."

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)

    retrieved_chunks = [chunks[i] for i in I[0]]

    # simple answer (student-level logic)
    answer = " ".join(retrieved_chunks)
    return answer
