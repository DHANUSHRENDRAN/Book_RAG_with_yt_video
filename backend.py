"""
BookRAG Backend — FastAPI + ChromaDB + SentenceTransformers + Groq
With hybrid retrieval (BM25 + semantic search), dynamic chunk sizing,
and content-type filtering.
Python 3.12 compatible
"""

import os
import re
import json
import uvicorn
from pathlib import Path
from typing import Optional

import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
UPLOAD_TEMP = "./uploads_temp"
Path(UPLOAD_TEMP).mkdir(exist_ok=True)
Path(CHROMA_PATH).mkdir(exist_ok=True)

# ─── Load Embedding Model (once at startup) ────────────────────────────────────
print("\n→ Loading embedding model (all-MiniLM-L6-v2)...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model ready!\n")

# ─── ChromaDB ──────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ─── Dynamic Chunk Size Tiers (mirrors rag.py) ───────────────────────────────
CHUNK_TIERS = [
    # (max_pages, chunk_size, chunk_overlap)
    (20, 500, 60),
    (60, 800, 100),
    (150, 1100, 140),
    (float("inf"), 1400, 180),
]


def get_pdf_splitter(total_pages: int) -> RecursiveCharacterTextSplitter:
    """Return a splitter whose chunk size scales with document page count."""
    for max_pages, chunk_size, overlap in CHUNK_TIERS:
        if total_pages <= max_pages:
            print(
                f"  → Dynamic chunking: {total_pages} pages → chunk_size={chunk_size}, overlap={overlap}"
            )
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )
    return RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=180)


# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BookRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def get_or_create_collection(name: str):
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^\s*[\-—–]*\s*\d+\s*[\-—–]*\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> dict:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text()
            if raw and raw.strip():
                cleaned = clean_text(raw)
                if len(cleaned) > 50:
                    pages.append({"page": i + 1, "text": cleaned})
    return {"pages": pages, "total_pages": total_pages}


def build_chunks_with_meta(
    pages: list,
    book_name: str,
    content_type: str = "pdf",
    total_pages: int = 0,
) -> tuple[list, list, list]:
    """
    Chunk all pages with DYNAMIC sizing based on total_pages (for PDFs).
    total_pages = 0 falls back to len(pages).
    """
    all_chunks: list[str] = []
    all_metas: list[dict] = []
    all_ids: list[str] = []

    splitter = get_pdf_splitter(total_pages or len(pages))

    for page_data in pages:
        page_num = page_data["page"]
        chunks = splitter.split_text(page_data["text"])
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metas.append(
                {
                    "book": book_name,
                    "page": page_num,
                    "chunk_index": j,
                    "content_type": content_type,
                }
            )
            all_ids.append(f"{book_name}__p{page_num}__c{j}")

    return all_chunks, all_metas, all_ids


# ──────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ──────────────────────────────────────────────────────────────────────────────

# ── Collections ────────────────────────────────────────────────────────────────


@app.get("/api/collections")
def list_collections():
    cols = chroma_client.list_collections()
    result = []
    for col_obj in cols:
        try:
            c = chroma_client.get_collection(col_obj.name)
            result.append({"name": col_obj.name, "count": c.count()})
        except Exception:
            pass
    return result


@app.post("/api/collections")
async def create_collection(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Collection name is required")
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", name):
        raise HTTPException(
            400, "Name must be 3-50 chars, only letters / numbers / _ / -"
        )
    col = get_or_create_collection(name)
    return {"name": name, "count": col.count()}


@app.delete("/api/collections/{name}")
def delete_collection(name: str):
    try:
        chroma_client.delete_collection(name)
        return {"deleted": name}
    except Exception as e:
        raise HTTPException(404, f"Collection not found: {e}")


# ── Books ──────────────────────────────────────────────────────────────────────


@app.get("/api/collections/{collection}/books")
def list_books(collection: str):
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
    try:
        col = chroma_client.get_collection(collection)
    except Exception:
        raise HTTPException(404, f"Collection '{collection}' not found")

    existing = col.get(where={"book": book_name})
    if not existing["ids"]:
        raise HTTPException(404, f"Book '{book_name}' not found in collection")
    col.delete(ids=existing["ids"])
    return {"deleted": book_name, "chunks_removed": len(existing["ids"])}


# ── Upload ─────────────────────────────────────────────────────────────────────


@app.post("/api/upload")
async def upload_book(
    file: UploadFile = File(...),
    collection: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    book_name = Path(file.filename).stem
    book_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", book_name).strip()
    book_name = re.sub(r"\s+", "_", book_name)[:80]

    temp_path = Path(UPLOAD_TEMP) / file.filename
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        extracted = extract_pdf_text(temp_path)
        pages = extracted["pages"]
        total_pages = extracted["total_pages"]

        if not pages:
            raise HTTPException(400, "Could not extract readable text from this PDF")

        # ── Dynamic chunk sizing based on total_pages ─────────────────────────
        chunks, metas, ids = build_chunks_with_meta(
            pages,
            book_name,
            content_type="pdf",
            total_pages=total_pages,  # ← key change
        )

        if not chunks:
            raise HTTPException(400, "No text chunks could be created from this PDF")

        col = get_or_create_collection(collection)

        try:
            existing = col.get(where={"book": book_name})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        BATCH = 64
        for i in range(0, len(chunks), BATCH):
            bc = chunks[i : i + BATCH]
            bm = metas[i : i + BATCH]
            bi = ids[i : i + BATCH]
            emb = EMBED_MODEL.encode(bc, show_progress_bar=False).tolist()
            col.add(documents=bc, embeddings=emb, ids=bi, metadatas=bm)

        return {
            "book": book_name,
            "total_pages": total_pages,
            "pages_with_text": len(pages),
            "total_chunks": len(chunks),
            "collection": collection,
            "chunk_size_used": (
                next(cs for mp, cs, _ in CHUNK_TIERS if total_pages <= mp)
            ),
        }

    finally:
        temp_path.unlink(missing_ok=True)


# ── Ask ────────────────────────────────────────────────────────────────────────


@app.post("/api/ask")
async def ask_question(request: Request):
    """Ask a question using hybrid retrieval (BM25 + semantic) with structured output."""
    from rag import ask as ask_rag

    body = await request.json()
    question: str = body.get("question", "").strip()
    collection: str = body.get("collection", "").strip()
    groq_key: str = body.get("groq_key", "").strip()
    n_results: int = int(body.get("n_results", 15))
    book_filter = body.get("book_filter")
    content_type_filter = body.get("content_type_filter")

    if not question:
        raise HTTPException(400, "Question is required")
    if not collection:
        raise HTTPException(400, "Select a collection first")
    if not groq_key:
        raise HTTPException(400, "Groq API key is required")

    try:
        result = ask_rag(
            question=question,
            collection_name=collection,
            groq_key=groq_key,
            n_results=n_results,
            book_filter=book_filter,
            content_type_filter=content_type_filter,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


# ─── Serve Frontend ────────────────────────────────────────────────────────────


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)


# """
# BookRAG Backend — FastAPI + ChromaDB + SentenceTransformers + Groq
# With hybrid retrieval (BM25 + semantic search) and content-type filtering
# Python 3.12 compatible
# """

# import os
# import re
# import json
# import uvicorn
# from pathlib import Path
# from typing import Optional

# import pdfplumber
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles

# import chromadb
# from sentence_transformers import SentenceTransformer
# from groq import Groq
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # ─── Config ────────────────────────────────────────────────────────────────────
# CHROMA_PATH = "./chroma_db"
# UPLOAD_TEMP = "./uploads_temp"
# Path(UPLOAD_TEMP).mkdir(exist_ok=True)
# Path(CHROMA_PATH).mkdir(exist_ok=True)

# # ─── Load Embedding Model (once at startup) ────────────────────────────────────
# print("\n→ Loading embedding model (all-MiniLM-L6-v2)...")
# EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# print("✓ Embedding model ready!\n")

# # ─── ChromaDB ──────────────────────────────────────────────────────────────────
# chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# # ─── Adaptive Text Splitter (same as rag.py for PDF) ────────────────────────────
# PDF_SPLITTER = RecursiveCharacterTextSplitter(
#     chunk_size=800,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
# )

# # ─── App ───────────────────────────────────────────────────────────────────────
# app = FastAPI(title="BookRAG API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ──────────────────────────────────────────────────────────────────────────────
# # HELPER FUNCTIONS
# # ──────────────────────────────────────────────────────────────────────────────


# def get_or_create_collection(name: str):
#     return chroma_client.get_or_create_collection(
#         name=name,
#         metadata={"hnsw:space": "cosine"},
#     )


# def clean_text(text: str) -> str:
#     """Remove junk from PDF text — headers, footers, extra whitespace."""
#     # Remove excessive whitespace
#     text = re.sub(r"\n{3,}", "\n\n", text)
#     text = re.sub(r" {2,}", " ", text)
#     # Remove page numbers like "— 42 —" or just lone numbers on a line
#     text = re.sub(r"^\s*[\-—–]*\s*\d+\s*[\-—–]*\s*$", "", text, flags=re.MULTILINE)
#     return text.strip()


# def extract_pdf_text(pdf_path: Path) -> dict:
#     """Extract text from PDF with page metadata."""
#     pages = []
#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)
#         for i, page in enumerate(pdf.pages):
#             raw = page.extract_text()
#             if raw and raw.strip():
#                 cleaned = clean_text(raw)
#                 if len(cleaned) > 50:  # skip near-empty pages
#                     pages.append(
#                         {
#                             "page": i + 1,
#                             "text": cleaned,
#                         }
#                     )
#     return {"pages": pages, "total_pages": total_pages}


# def build_chunks_with_meta(
#     pages: list, book_name: str, content_type: str = "pdf"
# ) -> tuple[list, list, list]:
#     """Chunk all pages with adaptive sizing based on content type."""
#     all_chunks = []
#     all_metas = []
#     all_ids = []

#     for page_data in pages:
#         page_num = page_data["page"]
#         chunks = PDF_SPLITTER.split_text(page_data["text"])
#         for j, chunk in enumerate(chunks):
#             chunk_id = f"{book_name}__p{page_num}__c{j}"
#             all_chunks.append(chunk)
#             all_metas.append(
#                 {
#                     "book": book_name,
#                     "page": page_num,
#                     "chunk_index": j,
#                     "content_type": content_type,
#                 }
#             )
#             all_ids.append(chunk_id)

#     return all_chunks, all_metas, all_ids


# # ──────────────────────────────────────────────────────────────────────────────
# # API ROUTES
# # ──────────────────────────────────────────────────────────────────────────────

# # ── Collections ────────────────────────────────────────────────────────────────


# @app.get("/api/collections")
# def list_collections():
#     cols = chroma_client.list_collections()
#     result = []
#     for col_obj in cols:
#         try:
#             c = chroma_client.get_collection(col_obj.name)
#             result.append({"name": col_obj.name, "count": c.count()})
#         except Exception:
#             pass
#     return result


# @app.post("/api/collections")
# async def create_collection(request: Request):
#     body = await request.json()
#     name = body.get("name", "").strip()
#     if not name:
#         raise HTTPException(400, "Collection name is required")
#     if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", name):
#         raise HTTPException(
#             400,
#             "Name must be 3-50 chars, only letters / numbers / _ / -",
#         )
#     col = get_or_create_collection(name)
#     return {"name": name, "count": col.count()}


# @app.delete("/api/collections/{name}")
# def delete_collection(name: str):
#     try:
#         chroma_client.delete_collection(name)
#         return {"deleted": name}
#     except Exception as e:
#         raise HTTPException(404, f"Collection not found: {e}")


# # ── Books ──────────────────────────────────────────────────────────────────────


# @app.get("/api/collections/{collection}/books")
# def list_books(collection: str):
#     try:
#         col = chroma_client.get_collection(collection)
#     except Exception:
#         raise HTTPException(404, f"Collection '{collection}' not found")

#     results = col.get(include=["metadatas"])
#     books: dict[str, dict] = {}
#     for meta in results["metadatas"]:
#         book = meta.get("book", "unknown")
#         if book not in books:
#             books[book] = {"chunks": 0, "pages": set()}
#         books[book]["chunks"] += 1
#         books[book]["pages"].add(meta.get("page", 0))

#     return [
#         {
#             "name": k,
#             "chunks": v["chunks"],
#             "pages": len(v["pages"]),
#         }
#         for k, v in sorted(books.items())
#     ]


# @app.delete("/api/collections/{collection}/books/{book_name}")
# def delete_book(collection: str, book_name: str):
#     try:
#         col = chroma_client.get_collection(collection)
#     except Exception:
#         raise HTTPException(404, f"Collection '{collection}' not found")

#     existing = col.get(where={"book": book_name})
#     if not existing["ids"]:
#         raise HTTPException(404, f"Book '{book_name}' not found in collection")
#     col.delete(ids=existing["ids"])
#     return {"deleted": book_name, "chunks_removed": len(existing["ids"])}


# # ── Upload ─────────────────────────────────────────────────────────────────────


# @app.post("/api/upload")
# async def upload_book(
#     file: UploadFile = File(...),
#     collection: str = Form(...),
# ):
#     if not file.filename or not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(400, "Only PDF files are supported")

#     # Sanitize book name
#     book_name = Path(file.filename).stem
#     book_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", book_name).strip()
#     book_name = re.sub(r"\s+", "_", book_name)[:80]

#     # Save temp
#     temp_path = Path(UPLOAD_TEMP) / file.filename
#     content = await file.read()
#     with open(temp_path, "wb") as f:
#         f.write(content)

#     try:
#         # Extract
#         extracted = extract_pdf_text(temp_path)
#         pages = extracted["pages"]
#         total_pages = extracted["total_pages"]

#         if not pages:
#             raise HTTPException(400, "Could not extract readable text from this PDF")

#         # Chunk with adaptive size (800 tokens) for PDFs
#         chunks, metas, ids = build_chunks_with_meta(
#             pages, book_name, content_type="pdf"
#         )

#         if not chunks:
#             raise HTTPException(400, "No text chunks could be created from this PDF")

#         # Get or create collection
#         col = get_or_create_collection(collection)

#         # Delete existing entries for this book (re-upload scenario)
#         try:
#             existing = col.get(where={"book": book_name})
#             if existing["ids"]:
#                 col.delete(ids=existing["ids"])
#         except Exception:
#             pass

#         # Embed in batches and store
#         BATCH = 64
#         for i in range(0, len(chunks), BATCH):
#             batch_chunks = chunks[i : i + BATCH]
#             batch_metas = metas[i : i + BATCH]
#             batch_ids = ids[i : i + BATCH]
#             embeddings = EMBED_MODEL.encode(
#                 batch_chunks, show_progress_bar=False
#             ).tolist()
#             col.add(
#                 documents=batch_chunks,
#                 embeddings=embeddings,
#                 ids=batch_ids,
#                 metadatas=batch_metas,
#             )

#         return {
#             "book": book_name,
#             "total_pages": total_pages,
#             "pages_with_text": len(pages),
#             "total_chunks": len(chunks),
#             "collection": collection,
#         }

#     finally:
#         temp_path.unlink(missing_ok=True)


# # ── Ask ────────────────────────────────────────────────────────────────────────


# @app.post("/api/ask")
# async def ask_question(request: Request):
#     """Ask a question using hybrid retrieval (BM25 + semantic)."""
#     from rag import ask as ask_rag

#     body = await request.json()
#     question: str = body.get("question", "").strip()
#     collection: str = body.get("collection", "").strip()
#     groq_key: str = body.get("groq_key", "").strip()
#     n_results: int = int(body.get("n_results", 15))
#     book_filter: Optional[str] = body.get("book_filter")
#     content_type_filter: Optional[str] = body.get("content_type_filter")

#     if not question:
#         raise HTTPException(400, "Question is required")
#     if not collection:
#         raise HTTPException(400, "Select a collection first")
#     if not groq_key:
#         raise HTTPException(400, "Groq API key is required")

#     try:
#         result = ask_rag(
#             question=question,
#             collection_name=collection,
#             groq_key=groq_key,
#             n_results=n_results,
#             book_filter=book_filter,
#             content_type_filter=content_type_filter,
#         )
#         return result
#     except ValueError as e:
#         raise HTTPException(400, str(e))
#     except LookupError as e:
#         raise HTTPException(404, str(e))
#     except Exception as e:
#         raise HTTPException(500, f"Error: {str(e)}")


# # ─── Serve Frontend ────────────────────────────────────────────────────────────


# @app.get("/")
# def serve_frontend():
#     return FileResponse("index.html")


# # ─── Run ───────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     uvicorn.run(
#         "backend:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=False,
#     )
