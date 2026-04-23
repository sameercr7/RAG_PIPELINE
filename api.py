from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json

from rag import answer
from db import get_admin_stats
from config import STATE_FILE

app = FastAPI(title="UP Police Matrix RAG", version="1.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

class SourceItem(BaseModel):
    name: str
    enabled: bool

class AdminConfig(BaseModel):
    ingestion_enabled: bool
    source_config: list[SourceItem]


@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<h1>UP Police Matrix RAG API</h1><p>index.html not found.</p>"


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    if os.path.exists("admin.html"):
        with open("admin.html", "r") as f:
            return f.read()
    return "<h1>Admin Page not found.</h1>"


@app.get("/api/admin/status")
def get_admin_status():
    stats = get_admin_stats()
    
    # Load state from file
    state = {"ingestion_enabled": False, "source_config": []}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            
    return {
        "stats": stats,
        "config": state
    }


@app.post("/api/admin/config")
def update_admin_config(cfg: AdminConfig):
    new_state = {
        "ingestion_enabled": cfg.ingestion_enabled,
        "source_config": [item.dict() for item in cfg.source_config]
    }
    with open(STATE_FILE, "w") as f:
        json.dump(new_state, f, indent=4)
    return {"message": "Configuration updated successfully"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        result = answer(req.question)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG error: {str(e)}")

    return ChatResponse(answer=result["answer"], sources=result["sources"])