"""
app.py — BookRAG  FastAPI entry point
════════════════════════════════════════════════════════════

Run with:
    python app.py
    — or —
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

All RAG logic lives in rag.py.  This file only contains:
  • FastAPI app + CORS setup
  • API routes (thin wrappers around rag.py functions)
  • Static file / index.html serving
  • uvicorn launcher
"""

import re
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ── Import all RAG logic from the dedicated module ──────────────────────────────
import rag
from rag import (
    UPLOAD_TEMP,
    chroma_client,
    ingest_book,
    ask,
)

# ════════════════════════════════════════════════════════════════════════════════
# APP  SETUP
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="BookRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════════
# COLLECTIONS  ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/collections")
def list_collections():
    """List all collections and their chunk counts."""
    result = []
    for col_obj in chroma_client.list_collections():
        try:
            c = chroma_client.get_collection(col_obj.name)
            result.append({"name": col_obj.name, "count": c.count()})
        except Exception:
            pass
    return result


@app.post("/api/collections")
async def create_collection(request: Request):
    """Create a new named collection."""
    body = await request.json()
    name = body.get("name", "").strip()

    if not name:
        raise HTTPException(400, "Collection name is required")
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", name):
        raise HTTPException(400, "Name must be 3–50 chars: letters, digits, _ or -")

    col = rag.get_or_create_collection(name)
    return {"name": name, "count": col.count()}


@app.delete("/api/collections/{name}")
def delete_collection(name: str):
    """Delete a collection by name."""
    try:
        chroma_client.delete_collection(name)
        return {"deleted": name}
    except Exception as e:
        raise HTTPException(404, f"Collection not found: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# BOOKS  ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/collections/{collection}/books")
def list_books(collection: str):
    """List all books ingested into a collection."""
    try:
        col = chroma_client.get_collection(collection)
    except Exception:
        raise HTTPException(404, f"Collection '{collection}' not found")

    results = col.get(include=["metadatas"])
    books: dict[str, dict] = {}
    for meta in results["metadatas"]:
        book = meta.get("book", "unknown")
        if book not in books:
            books[book] = {"chunks": 0, "pages": set()}
        books[book]["chunks"] += 1
        books[book]["pages"].add(meta.get("page", 0))

    return [
        {"name": k, "chunks": v["chunks"], "pages": len(v["pages"])}
        for k, v in sorted(books.items())
    ]


@app.delete("/api/collections/{collection}/books/{book_name}")
def delete_book(collection: str, book_name: str):
    """Remove a single book and all its chunks from a collection."""
    try:
        col = chroma_client.get_collection(collection)
    except Exception:
        raise HTTPException(404, f"Collection '{collection}' not found")

    existing = col.get(where={"book": book_name})
    if not existing["ids"]:
        raise HTTPException(404, f"Book '{book_name}' not found in collection")

    col.delete(ids=existing["ids"])
    return {"deleted": book_name, "chunks_removed": len(existing["ids"])}


# ════════════════════════════════════════════════════════════════════════════════
# UPLOAD  ROUTE  (PDF → ChromaDB)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_book(
    file: UploadFile = File(...),
    collection: str  = Form(...),
):
    """
    Accept a PDF upload, extract text, chunk it, embed it,
    and store everything in the specified ChromaDB collection.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Sanitise book name from filename
    book_name = Path(file.filename).stem
    book_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", book_name).strip()
    book_name = re.sub(r"\s+", "_", book_name)[:80]

    # Save to temp disk location
    temp_path = Path(UPLOAD_TEMP) / file.filename
    temp_path.write_bytes(await file.read())

    try:
        summary = ingest_book(temp_path, book_name, collection)
        return summary
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        temp_path.unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# UPLOAD ROUTE (YouTube → ChromaDB)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload_yt")
async def upload_youtube(request: Request):
    """
    Accept multiple YouTube URLs and ingest them into ChromaDB.
    """
    body = await request.json()
    urls = body.get("urls", [])
    collection = body.get("collection", "").strip()
    
    if not urls or not isinstance(urls, list):
        raise HTTPException(400, "Please provide a list of YouTube URLs.")
    if not collection:
        raise HTTPException(400, "Collection name is required.")
        
    try:
        summary = rag.ingest_youtube_urls(urls, collection)
        return summary
    except ValueError as e:
        raise HTTPException(400, str(e))


# ════════════════════════════════════════════════════════════════════════════════
# ASK  ROUTE  (question → RAG → Groq → answer)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/ask")
async def ask_question(request: Request):
    """
    Body (JSON):
      question    — user's question (required)
      collection  — collection name to query (required)
      groq_key    — Groq API key (required)
      n_results   — number of chunks to retrieve (default: 6)
      book_filter — optional book name to restrict retrieval
    """
    body        = await request.json()
    question    : str           = body.get("question",    "").strip()
    collection  : str           = body.get("collection",  "").strip()
    groq_key    : str           = body.get("groq_key",    "").strip()
    n_results   : int           = int(body.get("n_results",  6))
    book_filter : Optional[str] = body.get("book_filter")

    try:
        result = ask(
            question        = question,
            collection_name = collection,
            groq_key        = groq_key,
            n_results       = n_results,
            book_filter     = book_filter,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except LookupError as e:
        raise HTTPException(404, str(e))


# ════════════════════════════════════════════════════════════════════════════════
# FRONTEND  SERVE
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/")
def serve_frontend():
    """Serve the single-page frontend."""
    return FileResponse("index.html")


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY  POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 BookRAG is running!")
    print("   Open in browser → http://localhost:8000\n")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",   # listen on all interfaces (DO NOT open this in browser)
        port=8000,
        reload=False,
    )
