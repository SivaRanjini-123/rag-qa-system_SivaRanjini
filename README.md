# RAG-Based Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) based Question Answering system using FastAPI and FAISS.

## Features
- Upload PDF documents
- Convert documents into embeddings using Sentence Transformers
- Store embeddings using FAISS vector database
- Answer user questions based on uploaded documents

## Tech Stack
- Python
- FastAPI
- FAISS
- Sentence Transformers

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Start the server:
   python -m uvicorn app.main:app --reload
3. Open browser:
   http://127.0.0.1:8000/docs
rag-qa-system/
└── app/
    ├── main.py
    ├── ingest.py
    ├── rag.py
    ├── vector_store.py
    ├── models.py
    └── __init__.py
