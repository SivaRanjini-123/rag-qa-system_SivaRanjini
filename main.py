from fastapi import FastAPI, UploadFile, BackgroundTasks
from pydantic import BaseModel
import os
import shutil
from app.ingest import load_document
from app.rag import answer_question

app = FastAPI(title="RAG QA System")

os.makedirs("data/uploads", exist_ok=True)

class Question(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/upload")
def upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(load_document, file_path)
    return {"message": "File uploaded successfully"}

@app.post("/ask")
def ask_question(question: Question):
    answer = answer_question(question.query)
    return {"answer": answer}
