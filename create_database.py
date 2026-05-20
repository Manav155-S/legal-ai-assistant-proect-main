
import os
import re
import sys
import json
import uuid
import hashlib
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    from langchain_chroma import Chroma          # v2.2: replaces langchain_community
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    sys.exit(
        f"[FATAL] Missing dependency: {e}\n"
        f"  pip install langchain-chroma langchain-huggingface chromadb"
    )

try:
    from rank_bm25 import BM25Okapi
    import pickle
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[WARN] rank_bm25 not installed — BM25 index will be skipped.")
    print("       pip install rank-bm25")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PDF_FOLDER_PATH = "legal_pdfs"
CHROMA_DB_PATH  = "legal_db"
BM25_INDEX_PATH = "legal_db/bm25_index.pkl"
INDEX_CACHE     = "indexed_files.json"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 200       
FETCH_BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# CHUNKING CONFIGURATION
# ---------------------------------------------------------------------------
# Parent chunks — large context windows fed to the LLM
PARENT_STATUTE_SIZE    = 2000
PARENT_JUDGMENT_SIZE   = 3000

# Child chunks — what gets embedded and retrieved
CHILD_STATUTE_SIZE     = 600
CHILD_STATUTE_OVERLAP  = 120

CHILD_JUDGMENT_SIZE    = 900
CHILD_JUDGMENT_OVERLAP = 180

# Safety cap — max child chunks produced from a single parent block.
# Prevents runaway memory use on malformed / no-break documents.
MAX_CHUNKS_PER_PARENT  = 500
MIN_CHUNK_CHARS  = 100     # drop chunks shorter than this
MIN_ALPHA_RATIO  = 0.45    # at least 45 % of chars must be alphabetic
MAX_DIGIT_RATIO  = 0.40    # reject chunks that are mostly numbers (tables of contents)


# ---------------------------------------------------------------------------
# KNOWLEDGE BASE READER
# ---------------------------------------------------------------------------
def parse_knowledge_base(kb_path: str) -> list[dict]:
    """
    Parse knowledge_base.txt (produced by process_pdfs.py) into
    a list of document dicts:
      { filename, doc_type, total_pages, text }
    Supports both the v1 and v2 header formats.
    """
    docs = []
    with open(kb_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split on DOCUMENT_START markers
    blocks = re.split(r"={20,}\s*DOCUMENT_START\s*", content)

    for block in blocks[1:]:   # skip leading content before first DOCUMENT_START
        meta_match = re.match(
            r"FILENAME:\s*(.+?)\n"
            r"TYPE:\s*(\w+)\n"
            r"(?:TOTAL_PAGES:\s*(\S+)\n)?"
            r"(?:CHAR_COUNT:\s*(\S+)\n)?",
            block,
        )
        if not meta_match:
            continue

        filename    = meta_match.group(1).strip()
        doc_type    = meta_match.group(2).strip()
        total_pages = meta_match.group(3) or "?"

        # Strip the header lines and the footer
        text_body = re.sub(r"^.*?\n={20,}\n\n", "", block, count=1, flags=re.DOTALL)
        text_body = re.sub(r"\n={20,}\s*DOCUMENT_END.*$", "", text_body, flags=re.DOTALL)
        text_body = text_body.strip()

        if text_body:
            docs.append({
                "filename":    filename,
                "doc_type":    doc_type,
                "total_pages": total_pages,
                "text":        text_body,
            })

    return docs


# ---------------------------------------------------------------------------
# LEGAL-AWARE SECTION SPLITTER
# ---------------------------------------------------------------------------
def split_on_headings(text: str) -> list[tuple[str, str]]:
    """
    Split text on [[HEADING]] markers produced by process_pdfs.py.
    Returns list of (heading, body) tuples.
    If no headings found, returns [("", full_text)].
    """
    parts = re.split(r"(\[\[HEADING\]\]\s*.+)", text)

    if len(parts) == 1:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body    = parts[0]

    for part in parts[1:]:
        if part.startswith("[[HEADING]]"):
            if current_body.strip():
                sections.append((current_heading, current_body.strip()))
            current_heading = part.replace("[[HEADING]]", "").strip()
            current_body    = ""
        else:
            current_body += part

    if current_body.strip():
        sections.append((current_heading, current_body.strip()))

    return sections if sections else [("", text)]


def extract_page_number(text_fragment: str) -> Optional[int]:
    """Pull the nearest [PAGE:N] marker from a text fragment."""
    match = re.search(r"\[PAGE:(\d+)\]", text_fragment)
    return int(match.group(1)) if match else None


def strip_page_markers(text: str) -> str:
    return re.sub(r"\[PAGE:\d+\]\s*", "", text)


# ---------------------------------------------------------------------------
# CHUNK QUALITY FILTER
# ---------------------------------------------------------------------------
def is_quality_chunk(text: str) -> bool:
    """Return True if a chunk passes quality thresholds."""
    stripped = text.strip()
    length   = len(stripped)

    if length < MIN_CHUNK_CHARS:
        return False

    alpha = sum(c.isalpha() for c in stripped)
    digit = sum(c.isdigit() for c in stripped)

    if alpha / length < MIN_ALPHA_RATIO:
        return False
    if digit / length > MAX_DIGIT_RATIO:
        return False

    return True


# ---------------------------------------------------------------------------
# PARENT-CHILD CHUNKER
# ---------------------------------------------------------------------------
def make_child_chunks(text: str, size: int, overlap: int) -> list[str]:
    """
    Sliding-window chunker that respects sentence / paragraph breaks.
    Tries to split at paragraph > sentence > word boundaries.

    v2.1 fix: guarantees forward progress on every iteration so the loop
    always terminates even when the text contains no natural break points.
    MAX_CHUNKS_PER_PARENT safety cap provides a secondary bound.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Walk backwards from `end` to find a clean break point
        split_at = end
        for sep in separators:
            idx = text.rfind(sep, start, end)
            if idx > start:
                split_at = idx + len(sep)
                break

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        next_start = split_at - overlap
        if next_start <= start:
            next_start = start + max(1, size // 2)
        start = next_start
        if len(chunks) >= MAX_CHUNKS_PER_PARENT:
            print(
                f"  [WARN] MAX_CHUNKS_PER_PARENT ({MAX_CHUNKS_PER_PARENT}) reached — "
                f"truncating parent chunk. Remaining text discarded."
            )
            break

    return [c for c in chunks if c]


def chunk_document(doc: dict) -> list[dict]:
    """
    Produce child chunks from a document dict.
    Each chunk carries:
      - text (cleaned, no page markers)
      - metadata: source, doc_type, section_heading, hierarchy_level,
                  page_start, chunk_index, parent_id, total_chunks
    """
    doc_type    = doc["doc_type"]
    filename    = doc["filename"]
    full_text   = doc["text"]

    parent_size = PARENT_JUDGMENT_SIZE if doc_type == "JUDGMENT" else PARENT_STATUTE_SIZE
    child_size  = CHILD_JUDGMENT_SIZE  if doc_type == "JUDGMENT" else CHILD_STATUTE_SIZE
    child_ovlp  = CHILD_JUDGMENT_OVERLAP if doc_type == "JUDGMENT" else CHILD_STATUTE_OVERLAP

    sections = split_on_headings(full_text)
    all_chunks: list[dict] = []
    chunk_global_idx = 0

    for heading, section_body in sections:
        if not section_body.strip():
            continue

        parent_chunks = make_child_chunks(section_body, parent_size, overlap=0)

        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id         = str(uuid.uuid4())
            parent_page       = extract_page_number(parent_text)
            parent_text_clean = strip_page_markers(parent_text)

            child_chunks = make_child_chunks(parent_text_clean, child_size, child_ovlp)

            for c_idx, child_text in enumerate(child_chunks):
                child_text = child_text.strip()

                if not is_quality_chunk(child_text):
                    continue

                effective_heading = heading or (
                    parent_text_clean.splitlines()[0][:80] if parent_text_clean else ""
                )

                all_chunks.append({
                    "text": child_text,
                    "metadata": {
                        "source":           filename,
                        "doc_type":         doc_type,
                        "section_heading":  effective_heading,
                        "hierarchy_level":  "child",
                        "parent_id":        parent_id,
                        "parent_preview":   parent_text_clean[:200],
                        "page_start":       parent_page,
                        "chunk_index":      chunk_global_idx,
                        "child_index":      c_idx,
                        "parent_index":     p_idx,
                    },
                })
                chunk_global_idx += 1

    total = len(all_chunks)
    for chunk in all_chunks:
        chunk["metadata"]["total_chunks"] = total

    return all_chunks


# ---------------------------------------------------------------------------
# HASH-BASED DEDUPLICATION
# ---------------------------------------------------------------------------
def dedup_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks whose normalised text is identical (MD5)."""
    seen: set[str] = set()
    unique: list[dict] = []
    dupes = 0

    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk["text"].lower()).strip()
        h   = hashlib.md5(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
        else:
            dupes += 1

    if dupes:
        print(f"    Dedup removed {dupes:,} duplicate chunk(s).")

    return unique


# ---------------------------------------------------------------------------
# BM25 INDEX BUILDER
# ---------------------------------------------------------------------------
def _fetch_all_chroma_chunks(db: Chroma) -> list[dict]:
    """
    Fetch every document from a ChromaDB collection in pages of
    FETCH_BATCH_SIZE to avoid SQLite's 'too many SQL variables' error.

    Returns list of { text, metadata } dicts.
    """
    # Step 1 — fetch all IDs cheaply (no document data, no variable explosion)
    all_ids: list[str] = db.get(include=[])["ids"]
    total = len(all_ids)
    print(f"  Total chunks in ChromaDB : {total:,}")

    chunks: list[dict] = []

    for i in tqdm(
        range(0, total, FETCH_BATCH_SIZE),
        desc="  Fetching corpus for BM25",
        unit="batch",
    ):
        batch_ids = all_ids[i : i + FETCH_BATCH_SIZE]
        result    = db.get(ids=batch_ids, include=["documents", "metadatas"])
        for text, meta in zip(result["documents"], result["metadatas"]):
            chunks.append({"text": text, "metadata": meta})

    return chunks


def build_bm25_index(db: Chroma, index_path: str):
    """
    Fetch the full corpus from ChromaDB (in batches) and build + persist
    a BM25 index over it.
    """
    if not BM25_AVAILABLE:
        return

    try:
        all_chunks = _fetch_all_chroma_chunks(db)
    except Exception as e:
        print(f"  [WARN] Could not fetch corpus from ChromaDB: {e}")
        return

    print(f"  Building BM25 index over {len(all_chunks):,} chunks...")

    def tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    tokenised_corpus = [tokenise(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenised_corpus)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25":   bm25,
            "chunks": [c["text"]     for c in all_chunks],
            "metas":  [c["metadata"] for c in all_chunks],
        }, f)

    print(f"  BM25 index saved → {index_path}")


# ---------------------------------------------------------------------------
# INDEX CACHE
# ---------------------------------------------------------------------------
def load_index_cache(db_path: str) -> set:
    cache_path = os.path.join(db_path, INDEX_CACHE)
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return set(json.load(f))
    return set()


def save_index_cache(db_path: str, indexed: set):
    os.makedirs(db_path, exist_ok=True)
    cache_path = os.path.join(db_path, INDEX_CACHE)
    with open(cache_path, "w") as f:
        json.dump(list(indexed), f, indent=2)


# ---------------------------------------------------------------------------
# CORE BUILDER
# ---------------------------------------------------------------------------
def create_advanced_vector_database(force_rebuild: bool = False):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    db_path    = os.path.join(script_dir, CHROMA_DB_PATH)
    kb_path    = os.path.join(script_dir, "knowledge_base.txt")
    bm25_path  = os.path.join(script_dir, BM25_INDEX_PATH)

    if not os.path.exists(kb_path):
        print(f"[ERROR] knowledge_base.txt not found at {kb_path}")
        print("  Run process_pdfs.py first to generate it.")
        return

    print(f"Parsing knowledge base: {kb_path}")
    all_docs = parse_knowledge_base(kb_path)

    if not all_docs:
        print("[ERROR] No documents found in knowledge_base.txt.")
        return

    print(f"  Parsed {len(all_docs)} document(s) from knowledge base.")

    already_indexed = set() if force_rebuild else load_index_cache(db_path)
    new_docs = [d for d in all_docs if d["filename"] not in already_indexed]

    if not new_docs:
        print("  All documents already indexed. Nothing to do.")
        print("    Run with force_rebuild=True to rebuild from scratch.")
        return

    print(f"  Already indexed : {len(already_indexed)} doc(s)")
    print(f"  New / pending   : {len(new_docs)} doc(s)\n")

    all_chunks: list[dict] = []
    doc_counts = {"STATUTE": 0, "JUDGMENT": 0, "UNKNOWN": 0}

    for doc in tqdm(new_docs, desc="Chunking documents"):
        tqdm.write(
            f"  Processing : {doc['filename']} "
            f"({len(doc['text']):,} chars, type={doc['doc_type']})"
        )

        chunks = chunk_document(doc)
        chunks = dedup_chunks(chunks)

        doc_type = doc["doc_type"]
        doc_counts[doc_type] = doc_counts.get(doc_type, 0) + 1

        tqdm.write(
            f"  [{doc_type:8s}] {doc['filename'][:55]:<55} "
            f"→ {len(chunks):>4} chunks"
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[ERROR] No usable chunks produced. Check your knowledge base.")
        return

    print(f"\n  Total chunks    : {len(all_chunks):,}")
    print(f"  Statutes        : {doc_counts.get('STATUTE', 0)} doc(s)")
    print(f"  Judgments       : {doc_counts.get('JUDGMENT', 0)} doc(s)")

    print(f"\nLoading embedding model: '{EMBEDDING_MODEL}'...")
    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    texts     = [c["text"]     for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Writing to ChromaDB in {total_batches} batch(es) → '{CHROMA_DB_PATH}'\n")

    db_exists = os.path.exists(db_path) and not force_rebuild

    if db_exists:
        print("  Opening existing DB for incremental update...")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_fn)
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Adding batches"):
            db.add_texts(
                texts=texts[i : i + BATCH_SIZE],
                metadatas=metadatas[i : i + BATCH_SIZE],
            )
    else:
        print("  Creating new DB...")
        db = Chroma.from_texts(
            texts=texts[:BATCH_SIZE],
            embedding=embedding_fn,
            metadatas=metadatas[:BATCH_SIZE],
            persist_directory=db_path,
        )
        for i in tqdm(range(BATCH_SIZE, len(texts), BATCH_SIZE), desc="Adding batches"):
            db.add_texts(
                texts=texts[i : i + BATCH_SIZE],
                metadatas=metadatas[i : i + BATCH_SIZE],
            )

    # NOTE: db.persist() intentionally removed — Chroma 0.4.x+ auto-persists.
    print("  ChromaDB updated.")

    print("\n  Rebuilding BM25 index over full corpus...")
    build_bm25_index(db, bm25_path)

    newly_indexed = {d["filename"] for d in new_docs}
    save_index_cache(db_path, already_indexed | newly_indexed)

    total_indexed = len(already_indexed) + len(newly_indexed)
    print(f"\n{'─'*60}")
    print(f"  Vector database ready!")
    print(f"    ChromaDB     : {db_path}")
    print(f"    BM25 index   : {bm25_path}")
    print(f"    Docs indexed : {total_indexed} total  ({len(newly_indexed)} new)")
    print(f"    Chunks added : {len(all_chunks):,} this run")
    print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# HYBRID RETRIEVAL HELPER  (import this in your RAG pipeline)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Combine dense (ChromaDB) and sparse (BM25) retrieval.

    Usage:
        retriever = HybridRetriever(db_path="legal_db")
        results   = retriever.retrieve("GPA property sale Supreme Court", k=5)
        for r in results:
            print(r["text"], r["metadata"])
    """

    def __init__(
        self,
        db_path:   str   = "legal_db",
        bm25_path: str   = "legal_db/bm25_index.pkl",
        model:     str   = EMBEDDING_MODEL,
        dense_k:   int   = 10,
        sparse_k:  int   = 10,
        alpha:     float = 0.6,    # weight for dense scores (1-alpha = BM25 weight)
    ):
        self.alpha    = alpha
        self.dense_k  = dense_k
        self.sparse_k = sparse_k

        print("Loading embedding model...")
        embedding_fn = HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.db = Chroma(persist_directory=db_path, embedding_function=embedding_fn)

        self.bm25        = None
        self.bm25_corpus = []
        self.bm25_metas  = []

        if BM25_AVAILABLE and os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            self.bm25        = data["bm25"]
            self.bm25_corpus = data["chunks"]
            self.bm25_metas  = data["metas"]
            print(f"BM25 index loaded ({len(self.bm25_corpus):,} chunks).")
        else:
            print("[WARN] BM25 index not found — dense-only retrieval.")

    def _tokenise(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve top-k chunks using reciprocal-rank fusion of dense + sparse.
        Returns list of dicts: { text, metadata, score }.
        """
        scores:    dict[str, float] = {}
        texts_map: dict[str, str]   = {}
        metas_map: dict[str, dict]  = {}
        RRF_K = 60

        dense_results = self.db.similarity_search_with_score(query, k=self.dense_k)
        for rank, (doc, _score) in enumerate(dense_results):
            key = hashlib.md5(doc.page_content.encode()).hexdigest()
            scores[key]    = scores.get(key, 0) + self.alpha * (1 / (RRF_K + rank + 1))
            texts_map[key] = doc.page_content
            metas_map[key] = doc.metadata

        if self.bm25:
            tokenised_query = self._tokenise(query)
            bm25_scores     = self.bm25.get_scores(tokenised_query)
            top_indices     = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:self.sparse_k]

            for rank, idx in enumerate(top_indices):
                text = self.bm25_corpus[idx]
                meta = self.bm25_metas[idx]
                key  = hashlib.md5(text.encode()).hexdigest()
                scores[key]    = scores.get(key, 0) + (1 - self.alpha) * (1 / (RRF_K + rank + 1))
                texts_map[key] = text
                metas_map[key] = meta

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {
                "text":     texts_map[key],
                "metadata": metas_map[key],
                "score":    round(score, 6),
            }
            for key, score in ranked
        ]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    create_advanced_vector_database(force_rebuild=False)