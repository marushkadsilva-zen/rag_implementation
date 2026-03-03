from fastapi import FastAPI
from pydantic import BaseModel
from retrieval_pipeline import ask_question

app = FastAPI()

class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    answer, docs = ask_question(query.question)

    return {
        "answer": answer,
        "sources": [doc.metadata.get("source") for doc in docs]
    }