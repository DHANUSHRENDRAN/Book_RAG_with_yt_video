"""
rag.py — BookRAG core logic
════════════════════════════════════════════════════════════

Everything RAG-related lives here:
  • PDF text extraction        (extract_pdf_text)
  • Text cleaning              (clean_text)
  • Dynamic adaptive chunking  (build_chunks_with_meta)
  • ChromaDB collection helper (get_or_create_collection)
  • Embedding + storing a book (ingest_book)
  • YouTube via Whisper        (ingest_youtube_urls) ← yt_dlp + faster-whisper
  • Hybrid retrieval pipeline  (retrieve_chunks) — BM25 + semantic
  • Querying + calling Groq    (ask)
"""

import os
import re
import tempfile
import urllib.parse
from pathlib import Path

import chromadb
import pdfplumber
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ─── Paths ──────────────────────────────────────────────────────────────────────
CHROMA_PATH   = "./chroma_db"
UPLOAD_TEMP   = "./uploads_temp"
WHISPER_CACHE = "./whisper_cache"   # downloaded audio lives here temporarily

Path(CHROMA_PATH).mkdir(exist_ok=True)
Path(UPLOAD_TEMP).mkdir(exist_ok=True)
Path(WHISPER_CACHE).mkdir(exist_ok=True)

# ─── Models (loaded once at import time) ────────────────────────────────────────
print("\n→ Loading embedding model (all-MiniLM-L6-v2)...")
EMBED_MODEL: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model ready!\n")

# Whisper model loaded lazily (only when first YouTube URL is ingested)
_WHISPER_MODEL = None

def get_whisper_model():
    """
    Load faster-whisper on first use.
    Uses 'base' model — good balance of speed/accuracy on CPU.
    Downloads ~150MB once to ~/.cache/huggingface/
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            print("\n→ Loading Whisper model (base)...")
            # cpu + int8 = fastest on machines without GPU
            _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
            print("✓ Whisper model ready!\n")
        except ImportError:
            raise ImportError(
                "faster-whisper not installed.\n"
                "Run:  pip install faster-whisper"
            )
    return _WHISPER_MODEL

# ─── ChromaDB client ────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ─── Dynamic Chunk Size Thresholds ──────────────────────────────────────────────
CHUNK_TIERS = [
    (20,         500,  60),   # Small doc  (<= 20 pages)
    (60,         800, 100),   # Medium doc (<= 60 pages)
    (150,       1100, 140),   # Large doc  (<= 150 pages)
    (float("inf"), 1400, 180),# Very large doc (> 150 pages)
]

YOUTUBE_CHUNK_SIZE    = 1200
YOUTUBE_CHUNK_OVERLAP = 150

# ─── LLM settings ───────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 2048
TEMPERATURE  = 0.3
EMBED_BATCH  = 64
DEFAULT_N_RESULTS = 15


# ─── Dynamic n_results scaling ──────────────────────────────────────────────────
def dynamic_n_results(total_chunks: int, base: int = 15) -> int:
    if total_chunks < 100:   return max(base, 8)
    elif total_chunks < 500: return max(base, 12)
    elif total_chunks < 2000:return max(base, 18)
    else:                    return max(base, 25)


# ─── System prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an intelligent reading assistant helping someone explore their personal library.
Relevant excerpts from their books and videos have been retrieved and are provided below.

HOW TO STRUCTURE YOUR ANSWER — follow this format exactly:

## [A concise, descriptive title for the answer]

**Overview**
Write 2–3 sentences summarising the core answer directly. No preamble like "According to the excerpts".

**Key Points**
Break the answer into 3–6 focused bullet points. Each bullet should be a complete, informative sentence.
- Start each bullet with a bold keyword or phrase, e.g. **Query Rewriting** — then explain it.

**Details**
Write 1–2 paragraphs expanding on the most important aspects. Synthesise across all sources.
You may cite inline naturally, e.g. "(page 12)" or "(around [05:30] in the video)".

**Sources Used**
List the books/pages referenced at the end, e.g.:
- 📄 Advanced RAG Techniques — pages 5, 10, 13
- 📺 Video Title — timestamps 02:10, 15:45

RULES:
- Never open with "Based on the provided text" or similar.
- Use bullet points only inside Key Points — elsewhere write in flowing prose.
- If the answer genuinely cannot be found in the excerpts, say so in one line under Overview and stop.
- Do NOT invent information — only use what is in the excerpts.
"""


# ════════════════════════════════════════════════════════════════════════════════
# PDF UTILS
# ════════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════════
# SPLITTER FACTORY
# ════════════════════════════════════════════════════════════════════════════════

def get_pdf_splitter(total_pages: int) -> RecursiveCharacterTextSplitter:
    for max_pages, chunk_size, overlap in CHUNK_TIERS:
        if total_pages <= max_pages:
            print(f"  → Dynamic chunking: {total_pages} pages → chunk_size={chunk_size}, overlap={overlap}")
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )
    return RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=180)


YOUTUBE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=YOUTUBE_CHUNK_SIZE,
    chunk_overlap=YOUTUBE_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)


# ════════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ════════════════════════════════════════════════════════════════════════════════

def build_chunks_with_meta(
    pages: list[dict],
    book_name: str,
    content_type: str = "pdf",
    total_pages: int | None = None,
) -> tuple[list[str], list[dict], list[str]]:
    chunks, metas, ids = [], [], []
    splitter = YOUTUBE_SPLITTER if content_type == "youtube" else get_pdf_splitter(total_pages or len(pages))

    for page_data in pages:
        page_num    = page_data["page"]
        page_chunks = splitter.split_text(page_data["text"])
        for j, chunk in enumerate(page_chunks):
            chunks.append(chunk)
            metas.append({
                "book":         book_name,
                "page":         page_num,
                "chunk_index":  j,
                "content_type": content_type,
            })
            ids.append(f"{book_name}__p{page_num}__c{j}__{'yt' if content_type == 'youtube' else 'pdf'}")

    return chunks, metas, ids


# ════════════════════════════════════════════════════════════════════════════════
# CHROMADB HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def get_or_create_collection(name: str):
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ════════════════════════════════════════════════════════════════════════════════
# YOUTUBE — Download audio → Whisper transcription
# ════════════════════════════════════════════════════════════════════════════════

def _parse_video_id(url: str) -> str:
    """Extract YouTube video ID from any valid YouTube URL format."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/").split("?")[0]
    qs = urllib.parse.parse_qs(parsed.query)
    vid = qs.get("v", [""])[0]
    if vid:
        return vid
    # shorts format: /shorts/<id>
    m = re.search(r"/shorts/([^/?&]+)", parsed.path)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse YouTube video ID from: {url}")


def _download_audio(url: str, out_dir: Path) -> Path:
    """
    Download best audio from YouTube as .mp3 using yt_dlp.
    Returns path to downloaded file.
    """
    import yt_dlp

    video_id  = _parse_video_id(url)
    out_tmpl  = str(out_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "format":            "bestaudio/best",
        "outtmpl":           out_tmpl,
        "quiet":             True,
        "no_warnings":       True,
        "postprocessors": [{
            "key":            "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "96",   # lower quality = smaller file = faster
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title   = info.get("title",   f"YouTube Video ({video_id})")
        channel = info.get("uploader","Unknown Channel")
        duration= info.get("duration", 0)

    mp3_path = out_dir / f"{video_id}.mp3"
    if not mp3_path.exists():
        # Some systems save as .m4a etc — find whatever was saved
        candidates = list(out_dir.glob(f"{video_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"Audio download failed for {url}")
        mp3_path = candidates[0]

    return mp3_path, title, channel, duration


def _transcribe_with_whisper(audio_path: Path) -> list[dict]:
    """
    Transcribe audio file using faster-whisper.
    Returns list of segments: [{start, end, text}, ...]
    """
    model = get_whisper_model()
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="en",          # force English — change if needed
        vad_filter=True,        # skip silence
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(f"  → Whisper detected language: {info.language} (probability: {info.language_probability:.2f})")

    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text.strip(),
        })
    return result


def _segments_to_pages(segments: list[dict], seconds_per_page: int = 120) -> list[dict]:
    """
    Group transcript segments into "pages" of ~2 minutes each.
    This makes the chunk metadata meaningful (page = time block).
    Each page text includes timestamps like [MM:SS] for citations.
    """
    pages: list[dict] = []
    current_page_num  = 1
    current_start     = 0
    current_lines:list[str] = []

    for seg in segments:
        mm = int(seg["start"] // 60)
        ss = int(seg["start"] % 60)
        current_lines.append(f"[{mm:02d}:{ss:02d}] {seg['text']}")

        if seg["end"] - current_start >= seconds_per_page:
            pages.append({
                "page": current_page_num,
                "text": " ".join(current_lines),
            })
            current_page_num += 1
            current_start     = seg["end"]
            current_lines     = []

    if current_lines:
        pages.append({
            "page": current_page_num,
            "text": " ".join(current_lines),
        })

    return pages


def extract_youtube_info(url: str) -> dict:
    """
    Full pipeline: URL → download audio → Whisper transcription → pages.
    Falls back gracefully with clear error messages.
    """
    url       = url.strip()
    video_id  = _parse_video_id(url)
    audio_dir = Path(WHISPER_CACHE)

    print(f"\n→ Processing YouTube: {url}")

    # ── 1. Download audio ────────────────────────────────────────────────────
    print("  → Downloading audio (this may take 30–60 sec)...")
    try:
        audio_path, title, channel, duration = _download_audio(url, audio_dir)
    except Exception as e:
        raise ValueError(
            f"Could not download audio from {url}.\n"
            f"Reason: {e}\n"
            "Make sure yt_dlp and ffmpeg are installed."
        )

    print(f"  ✓ Downloaded: {title} ({duration//60:.0f} min)")

    # ── 2. Transcribe with Whisper ───────────────────────────────────────────
    print("  → Transcribing with Whisper (may take 1–3 min for long videos)...")
    try:
        segments = _transcribe_with_whisper(audio_path)
    except Exception as e:
        audio_path.unlink(missing_ok=True)
        raise ValueError(f"Whisper transcription failed: {e}")
    finally:
        # Clean up audio file — we only need the text
        audio_path.unlink(missing_ok=True)

    if not segments:
        raise ValueError(f"Whisper produced no transcript for {url}. Is it a silent video?")

    print(f"  ✓ Transcribed: {len(segments)} segments")

    # ── 3. Group into 2-min pages ────────────────────────────────────────────
    pages = _segments_to_pages(segments, seconds_per_page=120)

    return {
        "title":    title,
        "channel":  channel,
        "duration": duration,
        "pages":    pages,
        "url":      url,
    }


def ingest_youtube_urls(urls: list[str], collection_name: str) -> dict:
    """
    Ingest one or more YouTube URLs into a ChromaDB collection.
    Uses Whisper for transcription — works on ANY video, no captions needed.
    """
    col = get_or_create_collection(collection_name)
    total_videos = 0
    total_chunks = 0
    errors       = []

    for i, url in enumerate(urls):
        url = url.strip()
        if not url:
            continue

        try:
            yt_data = extract_youtube_info(url)
        except Exception as e:
            print(f"  ✗ Failed {url}: {e}")
            errors.append({"url": url, "error": str(e)})
            continue

        # Sanitize title → book_name
        book_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", yt_data["title"]).strip()
        book_name = re.sub(r"\s+", "_", book_name)[:80]
        if not book_name:
            book_name = f"video_{i}"

        # Chunk the pages
        chunks, metas, ids = build_chunks_with_meta(
            yt_data["pages"],
            book_name,
            content_type="youtube",
        )
        if not chunks:
            errors.append({"url": url, "error": "No chunks created"})
            continue

        # Remove old version if re-ingesting
        try:
            existing = col.get(where={"book": book_name})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        # Embed + store in batches
        for b in range(0, len(chunks), EMBED_BATCH):
            bc  = chunks[b : b + EMBED_BATCH]
            bm  = metas [b : b + EMBED_BATCH]
            bi  = ids   [b : b + EMBED_BATCH]
            emb = EMBED_MODEL.encode(bc, show_progress_bar=False).tolist()
            col.add(documents=bc, embeddings=emb, ids=bi, metadatas=bm)

        print(f"  ✓ Ingested '{book_name}': {len(chunks)} chunks")
        total_videos += 1
        total_chunks += len(chunks)

    if total_videos == 0:
        error_detail = "; ".join(e["error"] for e in errors) if errors else "Unknown error"
        raise ValueError(
            f"Could not ingest any YouTube URLs. Errors: {error_detail}"
        )

    return {
        "videos_ingested": total_videos,
        "total_chunks":    total_chunks,
        "collection":      collection_name,
        "errors":          errors,
    }


# ════════════════════════════════════════════════════════════════════════════════
# INGEST (PDF → ChromaDB)
# ════════════════════════════════════════════════════════════════════════════════

def ingest_book(pdf_path: Path, book_name: str, collection_name: str) -> dict:
    extracted   = extract_pdf_text(pdf_path)
    pages       = extracted["pages"]
    total_pages = extracted["total_pages"]

    if not pages:
        raise ValueError("Could not extract readable text from this PDF.")

    chunks, metas, ids = build_chunks_with_meta(
        pages, book_name, content_type="pdf", total_pages=total_pages
    )
    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    col = get_or_create_collection(collection_name)

    try:
        existing = col.get(where={"book": book_name})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    for i in range(0, len(chunks), EMBED_BATCH):
        bc  = chunks[i : i + EMBED_BATCH]
        bm  = metas [i : i + EMBED_BATCH]
        bi  = ids   [i : i + EMBED_BATCH]
        emb = EMBED_MODEL.encode(bc, show_progress_bar=False).tolist()
        col.add(documents=bc, embeddings=emb, ids=bi, metadatas=bm)

    return {
        "book":             book_name,
        "collection":       collection_name,
        "total_pages":      total_pages,
        "pages_with_text":  len(pages),
        "total_chunks":     len(chunks),
        "content_type":     "pdf",
    }


# ════════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL (BM25 + Semantic)
# ════════════════════════════════════════════════════════════════════════════════

def retrieve_chunks(
    question:             str,
    collection_name:      str,
    n_results:            int  = DEFAULT_N_RESULTS,
    book_filter:          str | None = None,
    content_type_filter:  str | None = None,
) -> tuple[list[str], list[dict], list[float]]:

    try:
        col = chroma_client.get_collection(collection_name)
    except Exception:
        raise LookupError(f"Collection '{collection_name}' not found.")

    total_count = col.count()
    if total_count == 0:
        raise ValueError("This collection is empty — upload some books first.")

    n_results = dynamic_n_results(total_count, base=n_results)

    # Semantic search
    q_vec = EMBED_MODEL.encode(question).tolist()
    query_kwargs: dict = {
        "query_embeddings": [q_vec],
        "n_results":        min(n_results * 3, total_count),
        "include":          ["documents", "metadatas", "distances"],
    }

    where_filters = []
    if book_filter:
        where_filters.append({"book": book_filter})
    if content_type_filter:
        where_filters.append({"content_type": content_type_filter})
    if where_filters:
        query_kwargs["where"] = (
            where_filters[0] if len(where_filters) == 1
            else {"$and": where_filters}
        )

    results          = col.query(**query_kwargs)
    semantic_chunks  = list(results["documents"][0])
    semantic_metas   = list(results["metadatas"][0])
    semantic_dists   = list(results["distances"][0])

    if not semantic_chunks:
        return [], [], []

    semantic_scores = [max(0, 1 - d) for d in semantic_dists]

    # BM25 scoring
    tokenized_docs = [c.lower().split() for c in semantic_chunks]
    query_tokens   = question.lower().split()
    try:
        bm25      = BM25Okapi(tokenized_docs)
        bm25_raw  = bm25.get_scores(query_tokens)
        max_bm25  = max(bm25_raw) if max(bm25_raw) > 0 else 1
        bm25_scores = [s / max_bm25 for s in bm25_raw]
    except Exception:
        bm25_scores = [0.5] * len(semantic_chunks)

    # Hybrid score: 70% semantic + 30% keyword
    hybrid_scores = [
        0.7 * sem + 0.3 * bm25
        for sem, bm25 in zip(semantic_scores, bm25_scores)
    ]

    combined = sorted(
        zip(semantic_chunks, semantic_metas, hybrid_scores),
        key=lambda x: x[2], reverse=True,
    )[:n_results]

    if book_filter and combined:
        combined.sort(key=lambda x: x[1].get("chunk_index", 0))

    if not combined:
        return [], [], []

    final_chunks, final_metas, final_scores = zip(*combined)
    return list(final_chunks), list(final_metas), list(final_scores)


# ════════════════════════════════════════════════════════════════════════════════
# ASK
# ════════════════════════════════════════════════════════════════════════════════

def ask(
    question:            str,
    collection_name:     str,
    groq_key:            str,
    n_results:           int  = DEFAULT_N_RESULTS,
    book_filter:         str | None = None,
    content_type_filter: str | None = None,
) -> dict:

    if not question:      raise ValueError("Question is required.")
    if not collection_name: raise ValueError("Select a collection first.")
    if not groq_key:      raise ValueError("Groq API key is required.")

    raw_chunks, raw_metas, hybrid_scores = retrieve_chunks(
        question=question,
        collection_name=collection_name,
        n_results=n_results,
        book_filter=book_filter,
        content_type_filter=content_type_filter,
    )

    if not raw_chunks:
        raise ValueError("No relevant content found for your question.")

    def friendly_title(raw: str) -> str:
        name = raw.replace("_", " ")
        for suffix in (" PDFDrive", " pdfDrive", " PDF", " pdf"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.strip()

    context_parts: list[str] = []
    sources:       list[dict] = []

    for chunk, meta, score in zip(raw_chunks, raw_metas, hybrid_scores):
        raw_book     = meta.get("book",         "unknown")
        page         = meta.get("page",         "?")
        content_type = meta.get("content_type", "pdf")
        title        = friendly_title(raw_book)
        relevance    = round(score * 100, 1)

        tag = "[📺 VIDEO]" if content_type == "youtube" else "[📄 PDF]"
        context_parts.append(f'{tag} Excerpt from "{title}" (page/block {page})\n{chunk}')
        sources.append({
            "book":         raw_book,
            "book_title":   title,
            "page":         page,
            "relevance":    relevance,
            "content_type": content_type,
            "preview":      chunk[:300] + ("..." if len(chunk) > 300 else ""),
        })

    context = "\n\n".join(context_parts)

    user_message = (
        f"Here are the relevant excerpts from the library:\n\n"
        f"{context}\n\n"
        f"╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌\n"
        f"Question: {question}"
    )

    groq_client = Groq(api_key=groq_key)
    chat = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    answer      = chat.choices[0].message.content
    tokens_used = chat.usage.total_tokens if chat.usage else 0

    return {
        "answer":           answer,
        "sources":          sources,
        "model":            GROQ_MODEL,
        "chunks_used":      len(raw_chunks),
        "tokens_used":      tokens_used,
        "retrieval_method": "hybrid (BM25 + semantic)",
    }















#yt whishper not working 
# """
# rag.py — BookRAG core logic
# ════════════════════════════════════════════════════════════

# Everything RAG-related lives here:
#   • PDF text extraction        (extract_pdf_text)
#   • Text cleaning              (clean_text)
#   • Dynamic adaptive chunking  (build_chunks_with_meta) — scales with doc size!
#   • ChromaDB collection helper (get_or_create_collection)
#   • Embedding + storing a book (ingest_book)
#   • Hybrid retrieval pipeline  (retrieve_chunks) — BM25 + semantic!
#   • Querying + calling Groq    (ask)

# app.py imports from this module — no FastAPI code here.
# """

# import re
# import urllib.parse
# from pathlib import Path

# import chromadb
# import pdfplumber
# import yt_dlp
# from youtube_transcript_api import YouTubeTranscriptApi
# from groq import Groq
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi

# # ─── Paths ──────────────────────────────────────────────────────────────────────
# CHROMA_PATH = "./chroma_db"
# UPLOAD_TEMP = "./uploads_temp"

# Path(CHROMA_PATH).mkdir(exist_ok=True)
# Path(UPLOAD_TEMP).mkdir(exist_ok=True)

# # ─── Models (loaded once at import time) ────────────────────────────────────────
# print("\n→ Loading embedding model (all-MiniLM-L6-v2)...")
# EMBED_MODEL: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
# print("✓ Embedding model ready!\n")

# # ─── ChromaDB client ────────────────────────────────────────────────────────────
# chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# # ─── Dynamic Chunk Size Thresholds ───────────────────────────────────────────────
# # These tune how chunks scale with document size (by page count)
# CHUNK_TIERS = [
#     # (max_pages, chunk_size, chunk_overlap)  — matched top-down
#     (20, 500, 60),  # Small doc  (<= 20 pages)  → tight, precise chunks
#     (60, 800, 100),  # Medium doc (<= 60 pages)  → balanced
#     (150, 1100, 140),  # Large doc  (<= 150 pages) → wider context
#     (float("inf"), 1400, 180),  # Very large doc (> 150 pages) → broadest sweep
# ]

# # YouTube always uses larger chunks to preserve conversation flow
# YOUTUBE_CHUNK_SIZE = 1200
# YOUTUBE_CHUNK_OVERLAP = 150

# # ─── LLM settings ───────────────────────────────────────────────────────────────
# GROQ_MODEL = "llama-3.3-70b-versatile"
# MAX_TOKENS = 2048
# TEMPERATURE = 0.3
# EMBED_BATCH = 64
# DEFAULT_N_RESULTS = 15


# # ─── Dynamic n_results scaling ──────────────────────────────────────────────────
# def dynamic_n_results(total_chunks: int, base: int = 15) -> int:
#     """Scale how many chunks to retrieve based on collection size."""
#     if total_chunks < 100:
#         return max(base, 8)
#     elif total_chunks < 500:
#         return max(base, 12)
#     elif total_chunks < 2000:
#         return max(base, 18)
#     else:
#         return max(base, 25)


# # ─── System prompt ───────────────────────────────────────────────────────────────
# SYSTEM_PROMPT = """\
# You are an intelligent reading assistant helping someone explore their personal library.
# Relevant excerpts from their books and videos have been retrieved and are provided below.

# HOW TO STRUCTURE YOUR ANSWER — follow this format exactly:

# ## [A concise, descriptive title for the answer]

# **Overview**
# Write 2–3 sentences summarising the core answer directly. No preamble like "According to the excerpts".

# **Key Points**
# Break the answer into 3–6 focused bullet points. Each bullet should be a complete, informative sentence.
# - Start each bullet with a bold keyword or phrase, e.g. **Query Rewriting** — then explain it.

# **Details**
# Write 1–2 paragraphs expanding on the most important aspects. Synthesise across all sources.
# You may cite inline naturally, e.g. "(page 12)" or "(around [05:30] in the video)".

# **Sources Used**
# List the books/pages referenced at the end, e.g.:
# - 📄 Advanced RAG Techniques — pages 5, 10, 13

# RULES:
# - Never open with "Based on the provided text" or similar.
# - Use bullet points only inside Key Points — elsewhere write in flowing prose.
# - If the answer genuinely cannot be found in the excerpts, say so in one line under Overview and stop.
# - Do NOT invent information — only use what is in the excerpts.
# """


# # ════════════════════════════════════════════════════════════════════════════════
# # PDF  UTILS
# # ════════════════════════════════════════════════════════════════════════════════


# def clean_text(text: str) -> str:
#     text = re.sub(r"\n{3,}", "\n\n", text)
#     text = re.sub(r" {2,}", " ", text)
#     text = re.sub(r"^\s*[\-—–]*\s*\d+\s*[\-—–]*\s*$", "", text, flags=re.MULTILINE)
#     return text.strip()


# def extract_pdf_text(pdf_path: Path) -> dict:
#     pages = []
#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)
#         for i, page in enumerate(pdf.pages):
#             raw = page.extract_text()
#             if raw and raw.strip():
#                 cleaned = clean_text(raw)
#                 if len(cleaned) > 50:
#                     pages.append({"page": i + 1, "text": cleaned})
#     return {"pages": pages, "total_pages": total_pages}


# # ════════════════════════════════════════════════════════════════════════════════
# # DYNAMIC SPLITTER FACTORY
# # ════════════════════════════════════════════════════════════════════════════════


# def get_pdf_splitter(total_pages: int) -> RecursiveCharacterTextSplitter:
#     """
#     Return a splitter whose chunk size scales with document length.
#     Larger documents get bigger chunks so retrieval stays broad enough
#     to surface relevant content spread across many pages.
#     """
#     for max_pages, chunk_size, overlap in CHUNK_TIERS:
#         if total_pages <= max_pages:
#             print(
#                 f"  → Dynamic chunking: {total_pages} pages → chunk_size={chunk_size}, overlap={overlap}"
#             )
#             return RecursiveCharacterTextSplitter(
#                 chunk_size=chunk_size,
#                 chunk_overlap=overlap,
#                 separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
#             )
#     # Fallback (should never reach here due to inf tier)
#     return RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=180)


# YOUTUBE_SPLITTER = RecursiveCharacterTextSplitter(
#     chunk_size=YOUTUBE_CHUNK_SIZE,
#     chunk_overlap=YOUTUBE_CHUNK_OVERLAP,
#     separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
# )


# # ════════════════════════════════════════════════════════════════════════════════
# # CHUNKING
# # ════════════════════════════════════════════════════════════════════════════════


# def build_chunks_with_meta(
#     pages: list[dict],
#     book_name: str,
#     content_type: str = "pdf",
#     total_pages: int | None = None,
# ) -> tuple[list[str], list[dict], list[str]]:
#     """
#     Split pages into chunks.
#     - PDFs:    dynamically sized based on total_pages
#     - YouTube: fixed larger size to preserve conversation flow
#     """
#     chunks, metas, ids = [], [], []

#     if content_type == "youtube":
#         splitter = YOUTUBE_SPLITTER
#     else:
#         # Use total_pages for dynamic sizing; fall back to len(pages) if not given
#         splitter = get_pdf_splitter(total_pages or len(pages))

#     for page_data in pages:
#         page_num = page_data["page"]
#         page_chunks = splitter.split_text(page_data["text"])

#         for j, chunk in enumerate(page_chunks):
#             chunks.append(chunk)
#             metas.append(
#                 {
#                     "book": book_name,
#                     "page": page_num,
#                     "chunk_index": j,
#                     "content_type": content_type,
#                 }
#             )
#             ids.append(
#                 f"{book_name}__p{page_num}__c{j}__{'yt' if content_type == 'youtube' else 'pdf'}"
#             )

#     return chunks, metas, ids


# # ════════════════════════════════════════════════════════════════════════════════
# # CHROMADB  HELPERS
# # ════════════════════════════════════════════════════════════════════════════════


# def get_or_create_collection(name: str):
#     return chroma_client.get_or_create_collection(
#         name=name,
#         metadata={"hnsw:space": "cosine"},
#     )


# # ════════════════════════════════════════════════════════════════════════════════
# # YOUTUBE UTILS
# # ════════════════════════════════════════════════════════════════════════════════


# def extract_youtube_info(url: str) -> dict:
#     parsed = urllib.parse.urlparse(url)
#     video_id = ""
#     if parsed.hostname == "youtu.be":
#         video_id = parsed.path[1:]
#     else:
#         video_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]

#     if not video_id:
#         raise ValueError(f"Could not parse YouTube URL: {url}")

#     transcript_text = ""
#     try:
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         try:
#             transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
#         except Exception:
#             transcript = list(transcript_list)[0]
#             if not transcript.language_code.startswith("en"):
#                 transcript = transcript.translate("en")

#         fetched = transcript.fetch()
#         parts = []
#         for t in fetched:
#             mm = int(t["start"] // 60)
#             ss = int(t["start"] % 60)
#             parts.append(f"[{mm:02d}:{ss:02d}] {t['text']}")
#         transcript_text = " ".join(parts)
#     except Exception as e:
#         print(f"Error fetching transcript for {video_id}: {e}")
#         raise ValueError(
#             f"Could not extract transcript for {url}. "
#             "Make sure the video has closed captions/subtitles enabled."
#         )

#     ydl_opts = {"quiet": True, "skip_download": True}
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         try:
#             info = ydl.extract_info(url, download=False)
#             title = info.get("title", f"YouTube Video ({video_id})")
#             description = info.get("description", "")
#             channel = info.get("uploader", "Unknown Channel")
#         except Exception:
#             title, description, channel = (
#                 f"YouTube Video ({video_id})",
#                 "",
#                 "Unknown Channel",
#             )

#     full_text = ""
#     if description:
#         full_text += f"Video Title: {title}\nChannel: {channel}\n\n--- VIDEO DESCRIPTION ---\n{description}\n\n"
#     if transcript_text:
#         full_text += f"--- TRANSCRIPT ---\n{transcript_text}\n"

#     if not full_text.strip():
#         raise ValueError(f"No text could be extracted from {url}")

#     return {"title": title, "text": full_text.strip(), "url": url}


# def build_youtube_chunks_with_meta(
#     yt_data: dict, book_name: str
# ) -> tuple[list[str], list[dict], list[str]]:
#     chunks, metas, ids = [], [], []
#     text_chunks = YOUTUBE_SPLITTER.split_text(yt_data["text"])
#     for j, chunk in enumerate(text_chunks):
#         chunks.append(chunk)
#         metas.append(
#             {
#                 "book": book_name,
#                 "page": 1,
#                 "chunk_index": j,
#                 "url": yt_data["url"],
#                 "content_type": "youtube",
#             }
#         )
#         ids.append(f"{book_name}__yt__c{j}")
#     return chunks, metas, ids


# def ingest_youtube_urls(urls: list[str], collection_name: str) -> dict:
#     col = get_or_create_collection(collection_name)
#     total_videos = 0
#     total_chunks = 0

#     for i, url in enumerate(urls):
#         url = url.strip()
#         if not url:
#             continue
#         try:
#             yt_data = extract_youtube_info(url)
#         except Exception as e:
#             print(f"Failed to extract {url}: {e}")
#             continue

#         book_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", yt_data["title"]).strip()
#         book_name = re.sub(r"\s+", "_", book_name)[:80]
#         if not book_name:
#             book_name = f"video_{i}"

#         chunks, metas, ids = build_youtube_chunks_with_meta(yt_data, book_name)
#         if not chunks:
#             continue

#         try:
#             existing = col.get(where={"book": book_name})
#             if existing["ids"]:
#                 col.delete(ids=existing["ids"])
#         except Exception:
#             pass

#         for b in range(0, len(chunks), EMBED_BATCH):
#             bc = chunks[b : b + EMBED_BATCH]
#             bm = metas[b : b + EMBED_BATCH]
#             bi = ids[b : b + EMBED_BATCH]
#             emb = EMBED_MODEL.encode(bc, show_progress_bar=False).tolist()
#             col.add(documents=bc, embeddings=emb, ids=bi, metadatas=bm)

#         total_videos += 1
#         total_chunks += len(chunks)

#     if total_videos == 0:
#         raise ValueError(
#             "Could not ingest any of the provided YouTube URLs. "
#             "Make sure they are valid and have captions/descriptions."
#         )
#     return {
#         "videos_ingested": total_videos,
#         "total_chunks": total_chunks,
#         "collection": collection_name,
#     }


# # ════════════════════════════════════════════════════════════════════════════════
# # INGEST  (PDF → ChromaDB)
# # ════════════════════════════════════════════════════════════════════════════════


# def ingest_book(pdf_path: Path, book_name: str, collection_name: str) -> dict:
#     """
#     Full pipeline: PDF → clean text → DYNAMIC chunks → embeddings → ChromaDB.
#     Chunk size scales automatically with the document's page count.
#     """
#     extracted = extract_pdf_text(pdf_path)
#     pages = extracted["pages"]
#     total_pages = extracted["total_pages"]

#     if not pages:
#         raise ValueError("Could not extract readable text from this PDF.")

#     # Pass total_pages so the splitter can pick the right tier
#     chunks, metas, ids = build_chunks_with_meta(
#         pages, book_name, content_type="pdf", total_pages=total_pages
#     )
#     if not chunks:
#         raise ValueError("No text chunks could be created from this PDF.")

#     col = get_or_create_collection(collection_name)

#     try:
#         existing = col.get(where={"book": book_name})
#         if existing["ids"]:
#             col.delete(ids=existing["ids"])
#     except Exception:
#         pass

#     for i in range(0, len(chunks), EMBED_BATCH):
#         bc = chunks[i : i + EMBED_BATCH]
#         bm = metas[i : i + EMBED_BATCH]
#         bi = ids[i : i + EMBED_BATCH]
#         emb = EMBED_MODEL.encode(bc, show_progress_bar=False).tolist()
#         col.add(documents=bc, embeddings=emb, ids=bi, metadatas=bm)

#     return {
#         "book": book_name,
#         "collection": collection_name,
#         "total_pages": total_pages,
#         "pages_with_text": len(pages),
#         "total_chunks": len(chunks),
#         "content_type": "pdf",
#     }


# # ════════════════════════════════════════════════════════════════════════════════
# # HYBRID RETRIEVAL (BM25 + Semantic Search)
# # ════════════════════════════════════════════════════════════════════════════════


# def retrieve_chunks(
#     question: str,
#     collection_name: str,
#     n_results: int = DEFAULT_N_RESULTS,
#     book_filter: str | None = None,
#     content_type_filter: str | None = None,
# ) -> tuple[list[str], list[dict], list[float]]:
#     """
#     Hybrid retrieval: BM25 (keyword) + semantic (vector) search.
#     n_results scales automatically with collection size.
#     """
#     try:
#         col = chroma_client.get_collection(collection_name)
#     except Exception:
#         raise LookupError(f"Collection '{collection_name}' not found.")

#     total_count = col.count()
#     if total_count == 0:
#         raise ValueError("This collection is empty — upload some books first.")

#     # Auto-scale n_results based on how large the collection is
#     n_results = dynamic_n_results(total_count, base=n_results)

#     # ── Step 1: Semantic search ────────────────────────────────────────────────
#     q_vec = EMBED_MODEL.encode(question).tolist()

#     query_kwargs: dict = {
#         "query_embeddings": [q_vec],
#         "n_results": min(n_results * 3, total_count),
#         "include": ["documents", "metadatas", "distances"],
#     }

#     where_filters = []
#     if book_filter:
#         where_filters.append({"book": book_filter})
#     if content_type_filter:
#         where_filters.append({"content_type": content_type_filter})

#     if where_filters:
#         query_kwargs["where"] = (
#             where_filters[0] if len(where_filters) == 1 else {"$and": where_filters}
#         )

#     results = col.query(**query_kwargs)

#     semantic_chunks = list(results["documents"][0])
#     semantic_metas = list(results["metadatas"][0])
#     semantic_dists = list(results["distances"][0])

#     if not semantic_chunks:
#         return [], [], []

#     semantic_scores = [max(0, 1 - d) for d in semantic_dists]

#     # ── Step 2: BM25 scoring ──────────────────────────────────────────────────
#     tokenized_docs = [c.lower().split() for c in semantic_chunks]
#     query_tokens = question.lower().split()

#     try:
#         bm25 = BM25Okapi(tokenized_docs)
#         bm25_raw = bm25.get_scores(query_tokens)
#         max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1
#         bm25_scores = [s / max_bm25 for s in bm25_raw]
#     except Exception:
#         bm25_scores = [0.5] * len(semantic_chunks)

#     # ── Step 3: Hybrid score (70 % semantic + 30 % keyword) ──────────────────
#     hybrid_scores = [
#         0.7 * sem + 0.3 * bm25 for sem, bm25 in zip(semantic_scores, bm25_scores)
#     ]

#     # ── Step 4: Re-rank, keep top n_results ──────────────────────────────────
#     combined = sorted(
#         zip(semantic_chunks, semantic_metas, hybrid_scores),
#         key=lambda x: x[2],
#         reverse=True,
#     )[:n_results]

#     # Preserve order when filtering by a single book
#     if book_filter and combined:
#         combined.sort(key=lambda x: x[1].get("chunk_index", 0))

#     if not combined:
#         return [], [], []

#     final_chunks, final_metas, final_scores = zip(*combined)
#     return list(final_chunks), list(final_metas), list(final_scores)


# # ════════════════════════════════════════════════════════════════════════════════
# # ASK  (question → Hybrid RAG → Groq → structured answer)
# # ════════════════════════════════════════════════════════════════════════════════


# def ask(
#     question: str,
#     collection_name: str,
#     groq_key: str,
#     n_results: int = DEFAULT_N_RESULTS,
#     book_filter: str | None = None,
#     content_type_filter: str | None = None,
# ) -> dict:
#     """
#     Retrieve relevant chunks with HYBRID search, then ask Groq for a
#     structured, heading-based answer.
#     """
#     if not question:
#         raise ValueError("Question is required.")
#     if not collection_name:
#         raise ValueError("Select a collection first.")
#     if not groq_key:
#         raise ValueError("Groq API key is required.")

#     raw_chunks, raw_metas, hybrid_scores = retrieve_chunks(
#         question=question,
#         collection_name=collection_name,
#         n_results=n_results,
#         book_filter=book_filter,
#         content_type_filter=content_type_filter,
#     )

#     if not raw_chunks:
#         raise ValueError(
#             "No relevant content found in this collection for your question."
#         )

#     def friendly_title(raw: str) -> str:
#         name = raw.replace("_", " ")
#         for suffix in (" PDFDrive", " pdfDrive", " PDF", " pdf"):
#             if name.endswith(suffix):
#                 name = name[: -len(suffix)]
#         return name.strip()

#     context_parts: list[str] = []
#     sources: list[dict] = []

#     for chunk, meta, score in zip(raw_chunks, raw_metas, hybrid_scores):
#         raw_book = meta.get("book", "unknown")
#         page = meta.get("page", "?")
#         content_type = meta.get("content_type", "pdf")
#         title = friendly_title(raw_book)
#         relevance = round(score * 100, 1)

#         source_hint = "[📺 VIDEO]" if content_type == "youtube" else "[📄 PDF]"
#         context_parts.append(
#             f'{source_hint} Excerpt from "{title}" (page {page})\n{chunk}'
#         )
#         sources.append(
#             {
#                 "book": raw_book,
#                 "book_title": title,
#                 "page": page,
#                 "relevance": relevance,
#                 "content_type": content_type,
#                 "preview": chunk[:300] + ("..." if len(chunk) > 300 else ""),
#             }
#         )

#     context = "\n\n".join(context_parts)

#     user_message = (
#         f"Here are the relevant excerpts from the library:\n\n"
#         f"{context}\n\n"
#         f"╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌\n"
#         f"Question: {question}"
#     )

#     groq_client = Groq(api_key=groq_key)
#     chat = groq_client.chat.completions.create(
#         model=GROQ_MODEL,
#         max_tokens=MAX_TOKENS,
#         temperature=TEMPERATURE,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_message},
#         ],
#     )

#     answer = chat.choices[0].message.content
#     tokens_used = chat.usage.total_tokens if chat.usage else 0

#     return {
#         "answer": answer,
#         "sources": sources,
#         "model": GROQ_MODEL,
#         "chunks_used": len(raw_chunks),
#         "tokens_used": tokens_used,
#         "retrieval_method": "hybrid (BM25 + semantic)",
#     }


# # this is before chaning working code but chuniking is not retrivbal not good
# # """
# # rag.py — BookRAG core logic
# # ════════════════════════════════════════════════════════════

# # Everything RAG-related lives here:
# #   • PDF text extraction        (extract_pdf_text)
# #   • Text cleaning              (clean_text)
# #   • Adaptive chunking          (build_chunks_with_meta) — now content-aware!
# #   • ChromaDB collection helper (get_or_create_collection)
# #   • Embedding + storing a book (ingest_book)
# #   • Hybrid retrieval pipeline  (retrieve_chunks) — BM25 + semantic!
# #   • Querying + calling Groq    (ask)

# # app.py imports from this module — no FastAPI code here.
# # """

# # import re
# # import urllib.parse
# # from pathlib import Path

# # import chromadb
# # import pdfplumber
# # import yt_dlp
# # from youtube_transcript_api import YouTubeTranscriptApi
# # from groq import Groq
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from sentence_transformers import SentenceTransformer
# # from rank_bm25 import BM25Okapi

# # # ─── Paths ──────────────────────────────────────────────────────────────────────
# # CHROMA_PATH  = "./chroma_db"
# # UPLOAD_TEMP  = "./uploads_temp"

# # Path(CHROMA_PATH).mkdir(exist_ok=True)
# # Path(UPLOAD_TEMP).mkdir(exist_ok=True)

# # # ─── Models (loaded once at import time) ────────────────────────────────────────
# # print("\n→ Loading embedding model (all-MiniLM-L6-v2)...")
# # EMBED_MODEL: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
# # print("✓ Embedding model ready!\n")

# # # ─── ChromaDB client ────────────────────────────────────────────────────────────
# # chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# # # ─── Adaptive Text Splitters (content-aware) ─────────────────────────────────────
# # # For PDFs: medium-sized chunks (800 tokens) — balance context vs specificity
# # PDF_SPLITTER = RecursiveCharacterTextSplitter(
# #     chunk_size=800,
# #     chunk_overlap=100,
# #     separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
# # )

# # # For YouTube transcripts: larger chunks (1200 tokens) — preserve conversation flow
# # YOUTUBE_SPLITTER = RecursiveCharacterTextSplitter(
# #     chunk_size=1200,
# #     chunk_overlap=150,
# #     separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
# # )

# # # ─── LLM settings ───────────────────────────────────────────────────────────────
# # GROQ_MODEL  = "llama-3.3-70b-versatile"
# # MAX_TOKENS  = 2000   # enough for a full, well-developed answer
# # TEMPERATURE = 0.35   # slightly warmer → more natural prose
# # EMBED_BATCH = 64
# # DEFAULT_N_RESULTS = 15  # Retrieve top 15 initially, then hybrid scoring

# # # ─── System prompt (controls answer quality decisively) ────────────────────────
# # SYSTEM_PROMPT = """\
# # You are an intelligent reading assistant helping someone explore their personal library.
# # Relevant excerpts from their books and videos have been retrieved and are provided below.

# # HOW TO WRITE YOUR ANSWER:
# # • Write in clear, flowing prose — like a knowledgeable friend who has read the material
# # • Synthesize information across ALL provided excerpts into ONE cohesive answer
# # • DO NOT open with "According to the excerpts", "Based on the provided text", or similar preambles — start directly with the answer
# # • Use bullet points ONLY if the user explicitly asks for a list; otherwise write in paragraphs
# # • You may cite sources naturally inline, e.g. "(page 12)" or "as noted on page 17"
# # • For videos, cite the timestamp if available, e.g. "(around [05:30] in the video)"
# # • Aim for 2–4 well-developed paragraphs that are thorough yet concise
# # • If the answer genuinely cannot be found in the excerpts, say so briefly in one sentence, then stop
# # • Do NOT invent information — only use what is in the excerpts below
# # """


# # # ════════════════════════════════════════════════════════════════════════════════
# # # PDF  UTILS
# # # ════════════════════════════════════════════════════════════════════════════════

# # def clean_text(text: str) -> str:
# #     """
# #     Strip common PDF noise:
# #       - 3+ consecutive blank lines → 2
# #       - double spaces
# #       - lone page numbers like "— 42 —"
# #     """
# #     text = re.sub(r"\n{3,}", "\n\n", text)
# #     text = re.sub(r" {2,}", " ", text)
# #     text = re.sub(r"^\s*[\-—–]*\s*\d+\s*[\-—–]*\s*$", "", text, flags=re.MULTILINE)
# #     return text.strip()


# # def extract_pdf_text(pdf_path: Path) -> dict:
# #     """
# #     Open a PDF and return a dict:
# #       {
# #         "pages":       [{"page": int, "text": str}, ...],
# #         "total_pages": int
# #       }
# #     Pages with fewer than 50 characters after cleaning are skipped.
# #     """
# #     pages = []
# #     with pdfplumber.open(pdf_path) as pdf:
# #         total_pages = len(pdf.pages)
# #         for i, page in enumerate(pdf.pages):
# #             raw = page.extract_text()
# #             if raw and raw.strip():
# #                 cleaned = clean_text(raw)
# #                 if len(cleaned) > 50:
# #                     pages.append({"page": i + 1, "text": cleaned})

# #     return {"pages": pages, "total_pages": total_pages}


# # # ════════════════════════════════════════════════════════════════════════════════
# # # CHUNKING (now adaptive!)
# # # ════════════════════════════════════════════════════════════════════════════════

# # def build_chunks_with_meta(
# #     pages: list[dict], book_name: str, content_type: str = "pdf"
# # ) -> tuple[list[str], list[dict], list[str]]:
# #     """
# #     Split pages into chunks with adaptive sizing based on content type.

# #     Args:
# #       pages       — list of {"page": int, "text": str}
# #       book_name   — name of the book/video
# #       content_type — "pdf" (smaller chunks) or "youtube" (larger chunks)

# #     Returns three parallel lists:
# #       chunks  — raw text of each chunk
# #       metas   — {"book": str, "page": int, "chunk_index": int, "content_type": str}
# #       ids     — unique ChromaDB document IDs
# #     """
# #     chunks, metas, ids = [], [], []

# #     # Use adaptive splitter based on content type
# #     splitter = YOUTUBE_SPLITTER if content_type == "youtube" else PDF_SPLITTER

# #     for page_data in pages:
# #         page_num     = page_data["page"]
# #         page_chunks  = splitter.split_text(page_data["text"])

# #         for j, chunk in enumerate(page_chunks):
# #             chunks.append(chunk)
# #             metas.append({
# #                 "book": book_name,
# #                 "page": page_num,
# #                 "chunk_index": j,
# #                 "content_type": content_type,
# #             })
# #             ids.append(f"{book_name}__p{page_num}__c{j}__{'yt' if content_type == 'youtube' else 'pdf'}")

# #     return chunks, metas, ids


# # # ════════════════════════════════════════════════════════════════════════════════
# # # CHROMADB  HELPERS
# # # ════════════════════════════════════════════════════════════════════════════════

# # def get_or_create_collection(name: str):
# #     """Return (or create) a ChromaDB collection using cosine similarity."""
# #     return chroma_client.get_or_create_collection(
# #         name=name,
# #         metadata={"hnsw:space": "cosine"},
# #     )


# # # ════════════════════════════════════════════════════════════════════════════════
# # # YOUTUBE UTILS
# # # ════════════════════════════════════════════════════════════════════════════════

# # def extract_youtube_info(url: str) -> dict:
# #     """Extracts transcript and description from a YouTube URL."""
# #     parsed = urllib.parse.urlparse(url)
# #     video_id = ""
# #     if parsed.hostname == 'youtu.be':
# #         video_id = parsed.path[1:]
# #     else:
# #         video_id = urllib.parse.parse_qs(parsed.query).get('v', [''])[0]

# #     if not video_id:
# #         raise ValueError(f"Could not parse YouTube URL: {url}")

# #     transcript_text = ""
# #     try:
# #         # Try native captions first (super fast)
# #         # Fetch the transcript list
# #         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

# #         # Try to find English, or just grab the first available one and translate it
# #         try:
# #             transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
# #         except Exception:
# #             # Fallback to the first available transcript (even auto-generated)
# #             transcript = list(transcript_list)[0]
# #             # Try to translate to English if it's not English
# #             if not transcript.language_code.startswith('en'):
# #                 transcript = transcript.translate('en')

# #         fetched = transcript.fetch()

# #         # Add basic timestamps inline for semantic richness
# #         transcript_parts = []
# #         for t in fetched:
# #             mm = int(t['start'] // 60)
# #             ss = int(t['start'] % 60)
# #             transcript_parts.append(f"[{mm:02d}:{ss:02d}] {t['text']}")
# #         transcript_text = " ".join(transcript_parts)
# #     except Exception as e:
# #         print(f"Error fetching transcript for {video_id}: {e}")
# #         raise ValueError(f"Could not extract transcript for {url}. Make sure the video has closed captions/subtitles enabled.")

# #     # Extract description and title
# #     ydl_opts = {"quiet": True, "skip_download": True}
# #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
# #         try:
# #             info = ydl.extract_info(url, download=False)
# #             title = info.get("title", f"YouTube Video ({video_id})")
# #             description = info.get("description", "")
# #             channel = info.get("uploader", "Unknown Channel")
# #         except Exception:
# #             title = f"YouTube Video ({video_id})"
# #             description = ""
# #             channel = "Unknown Channel"

# #     full_text = ""
# #     if description:
# #         full_text += f"Video Title: {title}\nChannel: {channel}\n\n--- VIDEO DESCRIPTION ---\n{description}\n\n"
# #     if transcript_text:
# #         full_text += f"--- TRANSCRIPT ---\n{transcript_text}\n"

# #     if not full_text.strip():
# #         raise ValueError(f"No text (transcript or description) could be extracted from {url}")

# #     return {"title": title, "text": full_text.strip(), "url": url}

# # def build_youtube_chunks_with_meta(yt_data: dict, book_name: str) -> tuple[list[str], list[dict], list[str]]:
# #     """Build chunks from YouTube transcript using larger chunk size for context preservation."""
# #     chunks, metas, ids = [], [], []
# #     text_chunks = YOUTUBE_SPLITTER.split_text(yt_data["text"])

# #     for j, chunk in enumerate(text_chunks):
# #         chunks.append(chunk)
# #         # Using 'page' as 1 so it fits the UI's expectation of page numbers
# #         metas.append({
# #             "book": book_name,
# #             "page": 1,
# #             "chunk_index": j,
# #             "url": yt_data["url"],
# #             "content_type": "youtube",
# #         })
# #         ids.append(f"{book_name}__yt__c{j}")

# #     return chunks, metas, ids

# # def ingest_youtube_urls(urls: list[str], collection_name: str) -> dict:
# #     """Ingests multiple YouTube URLs into a single collection."""
# #     col = get_or_create_collection(collection_name)

# #     total_videos = 0
# #     total_chunks = 0

# #     for i, url in enumerate(urls):
# #         url = url.strip()
# #         if not url: continue

# #         try:
# #             yt_data = extract_youtube_info(url)
# #         except Exception as e:
# #             print(f"Failed to extract {url}: {e}")
# #             continue

# #         book_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", yt_data["title"]).strip()
# #         book_name = re.sub(r"\s+", "_", book_name)[:80]
# #         if not book_name: book_name = f"video_{i}"

# #         chunks, metas, ids = build_youtube_chunks_with_meta(yt_data, book_name)
# #         if not chunks: continue

# #         # Remove old entries for this video
# #         try:
# #             existing = col.get(where={"book": book_name})
# #             if existing["ids"]:
# #                 col.delete(ids=existing["ids"])
# #         except Exception:
# #             pass

# #         # Embed and store
# #         for i in range(0, len(chunks), EMBED_BATCH):
# #             batch_chunks = chunks[i : i + EMBED_BATCH]
# #             batch_metas  = metas [i : i + EMBED_BATCH]
# #             batch_ids    = ids   [i : i + EMBED_BATCH]
# #             embeddings   = EMBED_MODEL.encode(batch_chunks, show_progress_bar=False).tolist()
# #             col.add(
# #                 documents=batch_chunks,
# #                 embeddings=embeddings,
# #                 ids=batch_ids,
# #                 metadatas=batch_metas,
# #             )

# #         total_videos += 1
# #         total_chunks += len(chunks)

# #     if total_videos == 0:
# #         raise ValueError("Could not ingest any of the provided YouTube URLs. Make sure they are valid and have captions/descriptions.")

# #     return {
# #         "videos_ingested": total_videos,
# #         "total_chunks": total_chunks,
# #         "collection": collection_name
# #     }


# # # ════════════════════════════════════════════════════════════════════════════════
# # # INGEST  (PDF → ChromaDB)
# # # ════════════════════════════════════════════════════════════════════════════════

# # def ingest_book(pdf_path: Path, book_name: str, collection_name: str) -> dict:
# #     """
# #     Full pipeline: PDF → clean text → chunks → embeddings → ChromaDB.

# #     Returns a summary dict with page / chunk counts.
# #     Raises ValueError if the PDF has no extractable text.
# #     """
# #     # 1. Extract
# #     extracted   = extract_pdf_text(pdf_path)
# #     pages       = extracted["pages"]
# #     total_pages = extracted["total_pages"]

# #     if not pages:
# #         raise ValueError("Could not extract readable text from this PDF.")

# #     # 2. Chunk (with content_type="pdf" for adaptive chunking)
# #     chunks, metas, ids = build_chunks_with_meta(pages, book_name, content_type="pdf")
# #     if not chunks:
# #         raise ValueError("No text chunks could be created from this PDF.")

# #     # 3. Get / create collection
# #     col = get_or_create_collection(collection_name)

# #     # 4. Remove old entries for this book (handles re-uploads cleanly)
# #     try:
# #         existing = col.get(where={"book": book_name})
# #         if existing["ids"]:
# #             col.delete(ids=existing["ids"])
# #     except Exception:
# #         pass

# #     # 5. Embed in batches and store
# #     for i in range(0, len(chunks), EMBED_BATCH):
# #         batch_chunks = chunks[i : i + EMBED_BATCH]
# #         batch_metas  = metas [i : i + EMBED_BATCH]
# #         batch_ids    = ids   [i : i + EMBED_BATCH]
# #         embeddings   = EMBED_MODEL.encode(batch_chunks, show_progress_bar=False).tolist()
# #         col.add(
# #             documents=batch_chunks,
# #             embeddings=embeddings,
# #             ids=batch_ids,
# #             metadatas=batch_metas,
# #         )

# #     return {
# #         "book":            book_name,
# #         "collection":      collection_name,
# #         "total_pages":     total_pages,
# #         "pages_with_text": len(pages),
# #         "total_chunks":    len(chunks),
# #         "content_type":    "pdf",
# #     }


# # # ════════════════════════════════════════════════════════════════════════════════
# # # HYBRID RETRIEVAL (BM25 + Semantic Search)
# # # ════════════════════════════════════════════════════════════════════════════════

# # def retrieve_chunks(
# #     question: str,
# #     collection_name: str,
# #     n_results: int = 15,
# #     book_filter: str | None = None,
# #     content_type_filter: str | None = None,  # "pdf" or "youtube"
# # ) -> tuple[list[str], list[dict], list[float]]:
# #     """
# #     Retrieve chunks using hybrid approach: BM25 (keyword) + semantic (vector) search.

# #     Strategy:
# #       1. Get all candidates from ChromaDB (semantic search)
# #       2. Build BM25 index from candidates
# #       3. Score each chunk with both metrics
# #       4. Rank by combined score (weighted average)
# #       5. Return top chunks in semantic relevance order for context

# #     Returns:
# #       chunks, metas, scores — parallel lists of retrieved chunks with combined scores
# #     """
# #     try:
# #         col = chroma_client.get_collection(collection_name)
# #     except Exception:
# #         raise LookupError(f"Collection '{collection_name}' not found.")

# #     total_count = col.count()
# #     if total_count == 0:
# #         raise ValueError("This collection is empty — upload some books first.")

# #     # ── Step 1: Semantic search to get candidates ──────────────────────────────
# #     q_vec = EMBED_MODEL.encode(question).tolist()

# #     # Build query filters
# #     query_kwargs = {
# #         "query_embeddings": [q_vec],
# #         "n_results": min(n_results * 3, total_count),  # Get 3x to allow re-ranking
# #         "include": ["documents", "metadatas", "distances"],
# #     }

# #     where_filters = []
# #     if book_filter:
# #         where_filters.append({"book": book_filter})
# #     if content_type_filter:
# #         where_filters.append({"content_type": content_type_filter})

# #     if where_filters:
# #         if len(where_filters) == 1:
# #             query_kwargs["where"] = where_filters[0]
# #         else:
# #             # Combine multiple filters with $and
# #             query_kwargs["where"] = {"$and": where_filters}

# #     results = col.query(**query_kwargs)

# #     semantic_chunks = list(results["documents"][0])
# #     semantic_metas  = list(results["metadatas"][0])
# #     semantic_dists  = list(results["distances"][0])

# #     if not semantic_chunks:
# #         return [], [], []

# #     # Convert distances to similarity scores (1 - distance for cosine)
# #     semantic_scores = [max(0, 1 - d) for d in semantic_dists]

# #     # ── Step 2: Build BM25 index and score chunks ──────────────────────────────
# #     # Tokenize documents for BM25
# #     tokenized_docs = [chunk.lower().split() for chunk in semantic_chunks]

# #     # Tokenize question
# #     query_tokens = question.lower().split()

# #     try:
# #         bm25 = BM25Okapi(tokenized_docs)
# #         bm25_scores = bm25.get_scores(query_tokens)
# #         # Normalize BM25 scores to [0, 1] range for fair weighting
# #         max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
# #         bm25_scores = [score / max_bm25 for score in bm25_scores]
# #     except Exception:
# #         # Fallback: equal weight if BM25 fails
# #         bm25_scores = [0.5] * len(semantic_chunks)

# #     # ── Step 3: Hybrid scoring (weighted combination) ──────────────────────────
# #     # Weight: 70% semantic + 30% keyword (semantic usually more reliable)
# #     hybrid_scores = [
# #         0.7 * sem_score + 0.3 * bm25_score
# #         for sem_score, bm25_score in zip(semantic_scores, bm25_scores)
# #     ]

# #     # ── Step 4: Re-rank by hybrid score, keep top n_results ───────────────────
# #     combined = list(zip(semantic_chunks, semantic_metas, hybrid_scores))
# #     combined.sort(key=lambda x: x[2], reverse=True)  # Sort by hybrid score
# #     combined = combined[:n_results]

# #     # ── Step 5: Re-order by chunk_index if filtering by book ─────────────────
# #     # This preserves chronological order for videos and logical flow for books
# #     if book_filter and combined:
# #         combined.sort(key=lambda x: x[1].get("chunk_index", 0))

# #     final_chunks, final_metas, final_scores = zip(*combined) if combined else ([], [], [])

# #     return list(final_chunks), list(final_metas), list(final_scores)


# # # ════════════════════════════════════════════════════════════════════════════════
# # # ASK  (question → Hybrid RAG → Groq → answer)
# # # ════════════════════════════════════════════════════════════════════════════════

# # def ask(
# #     question: str,
# #     collection_name: str,
# #     groq_key: str,
# #     n_results: int = 15,
# #     book_filter: str | None = None,
# #     content_type_filter: str | None = None,  # "pdf" or "youtube"
# # ) -> dict:
# #     """
# #     Retrieve relevant chunks using HYBRID search (BM25 + semantic), then ask Groq.

# #     Args:
# #       question          — user's question
# #       collection_name   — which collection to query
# #       groq_key          — Groq API key
# #       n_results         — number of top chunks to retrieve
# #       book_filter       — optional: filter to specific book/video
# #       content_type_filter — optional: "pdf" or "youtube" only

# #     Returns:
# #       {
# #         "answer":       str,
# #         "sources":      [{"book", "page", "relevance", "preview", "content_type"}, ...],
# #         "model":        str,
# #         "chunks_used":  int,
# #         "tokens_used":  int,
# #         "retrieval_method": "hybrid (BM25 + semantic)"
# #       }

# #     Raises ValueError for bad inputs, LookupError for missing collection.
# #     """
# #     # ── Validate ──────────────────────────────────────────────────────────────
# #     if not question:
# #         raise ValueError("Question is required.")
# #     if not collection_name:
# #         raise ValueError("Select a collection first.")
# #     if not groq_key:
# #         raise ValueError("Groq API key is required.")

# #     # ── Use hybrid retrieval ──────────────────────────────────────────────────
# #     raw_chunks, raw_metas, hybrid_scores = retrieve_chunks(
# #         question=question,
# #         collection_name=collection_name,
# #         n_results=n_results,
# #         book_filter=book_filter,
# #         content_type_filter=content_type_filter,
# #     )

# #     if not raw_chunks:
# #         raise ValueError("No relevant content found in this collection for your question.")

# #     # ── Helper: make an ugly filename-style name readable ────────────────────
# #     def friendly_title(raw: str) -> str:
# #         """'How_To_Win_Friends_PDFDrive' → 'How To Win Friends'"""
# #         name = raw.replace("_", " ")
# #         # Strip common PDF dump suffixes
# #         for suffix in (" PDFDrive", " pdfDrive", " PDF", " pdf"):
# #             if name.endswith(suffix):
# #                 name = name[: -len(suffix)]
# #         return name.strip()

# #     # ── Build context string and source list ──────────────────────────────────
# #     context_parts: list[str] = []
# #     sources:       list[dict] = []

# #     for chunk, meta, score in zip(raw_chunks, raw_metas, hybrid_scores):
# #         raw_book    = meta.get("book", "unknown")
# #         page        = meta.get("page", "?")
# #         content_type = meta.get("content_type", "pdf")
# #         title       = friendly_title(raw_book)
# #         relevance   = round(score * 100, 1)

# #         # Format each excerpt clearly for the LLM (with source type hint)
# #         source_hint = "[📺 VIDEO]" if content_type == "youtube" else "[📄 PDF]"
# #         context_parts.append(
# #             f'{source_hint} Excerpt from "{title}" (page {page})\n{chunk}'
# #         )
# #         sources.append({
# #             "book":         raw_book,
# #             "book_title":   title,
# #             "page":         page,
# #             "relevance":    relevance,
# #             "content_type": content_type,
# #             "preview":      chunk[:300] + ("..." if len(chunk) > 300 else ""),
# #         })

# #     context = "\n\n".join(context_parts)

# #     # ── Build the user message with clear framing ─────────────────────────────
# #     user_message = (
# #         f"Here are the relevant excerpts from the library:\n\n"
# #         f"{context}\n\n"
# #         f"╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌\n"
# #         f"Question: {question}"
# #     )

# #     # ── Call Groq ─────────────────────────────────────────────────────────────
# #     groq_client = Groq(api_key=groq_key)
# #     chat = groq_client.chat.completions.create(
# #         model=GROQ_MODEL,
# #         max_tokens=MAX_TOKENS,
# #         temperature=TEMPERATURE,
# #         messages=[
# #             {"role": "system", "content": SYSTEM_PROMPT},
# #             {"role": "user",   "content": user_message},
# #         ],
# #     )

# #     answer      = chat.choices[0].message.content
# #     tokens_used = chat.usage.total_tokens if chat.usage else 0

# #     return {
# #         "answer":             answer,
# #         "sources":            sources,
# #         "model":              GROQ_MODEL,
# #         "chunks_used":        len(raw_chunks),
# #         "tokens_used":        tokens_used,
# #         "retrieval_method":   "hybrid (BM25 + semantic)",
# #     }
