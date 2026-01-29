import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "data/faiss_index/index.faiss"
META_PATH = "data/faiss_index/chunks.pkl"

def save_embeddings(chunks):
    """
    Converts chunks into embeddings and stores them in FAISS
    """

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("data/faiss_index", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("[VECTOR] Embeddings stored successfully")


def load_index():
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks
