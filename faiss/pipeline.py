"""
hybrid_FAISS/pipeline.py

Single end-to-end pipeline for building a hybrid (dense + sparse) FAISS
index over Bengali fact-check articles.

Stages (run in order by main()):
    1. discover     - build target_urls.csv from Bengali fact-check sitemaps, falling
                      back to robots.txt / common sitemap paths if a site's configured
                      sitemap URL is missing or moved
    2. scrape       - download articles concurrently, extract title/date/body -> raw corpus
    3. clean        - normalise/dedupe/filter raw corpus (Bengali-script check) -> clean corpus
    4. chunk        - split clean corpus into overlapping text chunks
    5. embed+index  - BGE-M3 dense + sparse embeddings (GPU, tuned for a 6GB card),
                      then an IndexHNSWFlat (no training step required)

Everything reads/writes inside this one directory (hybrid_FAISS/) so the
whole project is self-contained:

    hybrid_FAISS/
        pipeline.py
        target_urls.csv
        output/
            factcheck_corpus.json          (raw scrape, one JSON object per line)
            factcheck_corpus_clean.json    (cleaned/deduped, feeds the indexer)
        logs/
            completed_urls.txt
            failed_urls.txt                (permanent failures - won't retry)
        checkpoint_data_hybrid/
            dense_vectors.npy              (embedding-phase checkpoint)
            factcheck_index.faiss          (IndexHNSWFlat, written once at the end)
            metadata.pkl
            sparse_weights.pkl
            checkpoint_state.pkl

Run:
    python pipeline.py                     # full pipeline
    python pipeline.py --skip-scrape       # reuse existing raw corpus, just clean+index
    python pipeline.py --only clean        # run a single stage
    python pipeline.py --workers 16        # more concurrent scraper threads
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import logging
import os
import pickle
import re
import threading
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import cloudscraper
import numpy as np
import pandas as pd
import requests
import trafilatura

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("hybrid_faiss")

# --------------------------------------------------------------------------
# Everything anchored to this directory, so it doesn't matter where the
# script is invoked from.
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

TARGETS_FILE = BASE_DIR / "target_urls.csv"
OUTPUT_DIR = BASE_DIR / "output"
RAW_CORPUS_FILE = OUTPUT_DIR / "factcheck_corpus.json"
CLEAN_CORPUS_FILE = OUTPUT_DIR / "factcheck_corpus_clean.json"

LOG_DIR = BASE_DIR / "logs"
COMPLETED_LOG = LOG_DIR / "completed_urls.txt"
FAILED_LOG = LOG_DIR / "failed_urls.txt"

CHECKPOINT_DIR = BASE_DIR / "checkpoint_data_hybrid"

# All sources are Bengali fact-check outlets. This pipeline only ever
# gathers Bengali-language facts - there is no English or mixed-language
# path anywhere below.
SITEMAPS = {
    "rumorscanner": "https://rumorscanner.com/sitemap_index.xml",
    "boombd": "https://bangla.boombd.com/sitemap.xml",
    "bdfactcheck": "https://bdfactcheck.com/sitemap.xml",
    "jachai": "https://jachai.org/sitemap.xml",
    "fact-watch": "https://www.fact-watch.org/sitemap.xml",
    "bangla-fact": "https://banglafact.com/sitemap.xml",
    "dismislab": "https://dismislab.com/sitemap.xml",
    "prothom-alo": "https://www.prothomalo.com/sitemap.xml"
}

# Fallback paths tried, in order, when a source's configured sitemap URL
# doesn't resolve (404 / moved / renamed). robots.txt is tried first since
# it's authoritative when present; these common paths are the backstop.
COMMON_SITEMAP_PATHS = [
    "/sitemap_index.xml",
    "/sitemap.xml",
    "/sitemap-index.xml",
    "/sitemap1.xml",
    "/post-sitemap.xml",
    "/news-sitemap.xml",
]

# Minimum fraction of characters that must fall in the Bengali Unicode block
# for an article to be kept. Filters out stray off-language pages that
# occasionally show up in these sitemaps.
BENGALI_RANGE = (0x0980, 0x09FF)
MIN_BENGALI_RATIO = 0.4
MIN_BODY_CHARS = 200

RETRY_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0

# Scraping is I/O-bound (waiting on remote servers), so a thread pool gives
# a near-linear speedup over the old one-URL-at-a-time loop. If a particular
# site starts returning 429s, drop --workers for that run rather than
# raising it further — the per-URL backoff above still applies per thread.
DEFAULT_MAX_WORKERS = 8

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "desktop": True}
)
# Widen the connection pool so concurrent threads aren't bottlenecked on it.
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=DEFAULT_MAX_WORKERS * 2, pool_maxsize=DEFAULT_MAX_WORKERS * 2
)
scraper.mount("https://", _adapter)
scraper.mount("http://", _adapter)


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# Stage 1: sitemap discovery
# ==========================================================================

def _parse_sitemap_bytes(raw: bytes, sitemap_url: str) -> list[str]:
    """Parses one sitemap payload, transparently handling gzip-compressed
    sitemaps (common on boombd/fact-watch/bangla-fact)."""
    if raw[:2] == b"\x1f\x8b" or sitemap_url.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                raw = gz.read()

    urls: list[str] = []
    tree = ET.fromstring(raw)
    namespaces = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    nested = tree.findall("ns:sitemap/ns:loc", namespaces)
    if nested:
        for loc in nested:
            if loc.text:
                log.info(f"Navigating nested sitemap: {loc.text}")
                urls.extend(fetch_sitemap_urls(loc.text))
        return urls

    pages = tree.findall("ns:url/ns:loc", namespaces)
    urls.extend([p.text for p in pages if p.text])
    return urls


def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Recursively parses XML (or gzipped XML) sitemaps to build a catalogue
    of article URLs."""
    try:
        response = scraper.get(sitemap_url, timeout=15)
        if response.status_code != 200:
            log.error(f"Failed to access sitemap {sitemap_url} (Status: {response.status_code})")
            return []
        return _parse_sitemap_bytes(response.content, sitemap_url)
    except Exception as e:
        log.error(f"Error parsing sitemap {sitemap_url}: {e}")
        return []


def _site_root(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _sitemaps_from_robots(root: str) -> list[str]:
    """Reads robots.txt and pulls out any declared 'Sitemap:' entries."""
    try:
        response = scraper.get(f"{root}/robots.txt", timeout=10)
        if response.status_code != 200:
            return []
        return [
            line.split(":", 1)[1].strip()
            for line in response.text.splitlines()
            if line.strip().lower().startswith("sitemap:")
        ]
    except Exception as e:
        log.warning(f"Could not read robots.txt at {root}: {e}")
        return []


def find_sitemap_urls(configured_url: str, source: str) -> list[str]:
    """Tries the configured sitemap URL first. If that comes back empty
    (moved, renamed, wrong path), falls back to whatever robots.txt
    declares, then a list of common sitemap paths on the same domain,
    before giving up on that source entirely."""
    urls = fetch_sitemap_urls(configured_url)
    if urls:
        return urls

    log.warning(f"No sitemap at {configured_url} for {source}; attempting discovery...")
    root = _site_root(configured_url)

    candidates = _sitemaps_from_robots(root)
    candidates += [root + path for path in COMMON_SITEMAP_PATHS]

    tried = {configured_url}
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.add(candidate)
        urls = fetch_sitemap_urls(candidate)
        if urls:
            log.info(f"Discovered working sitemap for {source} at {candidate}")
            return urls

    log.error(f"Could not find any sitemap for {source} (tried robots.txt + common paths).")
    return []


def build_target_catalogue() -> pd.DataFrame:
    """Generates the initial CSV of all target URLs if it does not already exist."""
    if TARGETS_FILE.exists():
        log.info(f"Target catalogue '{TARGETS_FILE}' found. Skipping discovery phase.")
        return pd.read_csv(TARGETS_FILE)

    log.info("Initiating sitemap discovery phase...")
    all_targets = []
    for source, url in SITEMAPS.items():
        log.info(f"Harvesting directory for {source}...")
        for u in find_sitemap_urls(url, source):
            all_targets.append({"source": source, "url": u})

    df_urls = pd.DataFrame(all_targets).drop_duplicates(subset=["url"])
    df_urls.to_csv(TARGETS_FILE, index=False)
    log.info(f"Discovery complete. Saved {len(df_urls)} unique URLs to {TARGETS_FILE}.")
    return df_urls


# ==========================================================================
# Stage 2: scraping (date of publication + article body, per-URL)
# ==========================================================================

def get_processed_urls(filepath: Path) -> set[str]:
    if not filepath.exists():
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def log_url(filepath: Path, url: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"{url}\n")


def scrape_article(url: str, source: str) -> tuple[str, dict | None]:
    """Downloads a page and extracts headline, publish date, and main body
    text. Returns (status, data) where status is one of:
        "ok"             - data is a populated dict
        "transient_fail" - worth retrying later (timeout, 5xx, rate limit)
        "permanent_fail" - dead link / no extractable content, don't retry
    """
    last_status = "transient_fail"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = scraper.get(url, timeout=10)

            if res.status_code == 200:
                extracted_text = trafilatura.extract(
                    res.text, include_comments=False, include_tables=False
                )
                metadata = trafilatura.extract_metadata(res.text)

                title_text = metadata.title.strip() if metadata and metadata.title else ""
                date_text = metadata.date.strip() if metadata and metadata.date else "Unknown Date"
                body_text = extracted_text.strip() if extracted_text else ""

                if not title_text or not body_text:
                    return "permanent_fail", None

                return "ok", {
                    "source": source,
                    "lang": "bn",
                    "url": url,
                    "title": title_text,
                    "date": date_text,
                    "text": body_text,
                }

            if res.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                log.warning(f"{url} -> HTTP {res.status_code}, retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if res.status_code == 404:
                return "permanent_fail", None

            last_status = "transient_fail"
            break

        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                log.warning(f"Failed to scrape {url}: {e} - retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            log.warning(f"Failed to scrape {url} after {MAX_RETRIES} attempts: {e}")
            last_status = "transient_fail"

    return last_status, None


def run_scrape_stage(max_workers: int = DEFAULT_MAX_WORKERS):
    ensure_dirs()
    df_targets = build_target_catalogue()

    completed_urls = get_processed_urls(COMPLETED_LOG)
    failed_urls = get_processed_urls(FAILED_LOG)

    todo = [
        (row["url"], row["source"])
        for _, row in df_targets.iterrows()
        if row["url"] not in completed_urls and row["url"] not in failed_urls
    ]

    log.info(
        f"Starting extraction loop. {len(completed_urls)} already processed, "
        f"{len(failed_urls)} permanently failed, {len(todo)} remaining. "
        f"Using {max_workers} concurrent workers."
    )

    write_lock = threading.Lock()
    done_count = 0

    with open(RAW_CORPUS_FILE, "a", encoding="utf-8") as corpus, \
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:

        future_to_target = {
            pool.submit(scrape_article, url, source): (url, source)
            for url, source in todo
        }

        for future in concurrent.futures.as_completed(future_to_target):
            url, source = future_to_target[future]
            done_count += 1
            try:
                status, data = future.result()
            except Exception as e:
                log.warning(f"Unhandled error scraping {url}: {e}")
                status, data = "transient_fail", None

            # Threads finish out of order and share the corpus file/logs,
            # so writes are serialised here — the network waits happen
            # concurrently above, only the (fast) file I/O is locked.
            with write_lock:
                if status == "ok":
                    corpus.write(json.dumps(data, ensure_ascii=False) + "\n")
                    corpus.flush()
                    log_url(COMPLETED_LOG, url)
                elif status == "permanent_fail":
                    log_url(FAILED_LOG, url)
                else:
                    log.info(f"Transient failure on {url}, will retry on next run.")

            if done_count % 25 == 0 or done_count == len(todo):
                log.info(f"[{done_count}/{len(todo)}] scraped")

    log.info("Scraping stage finished.")


# ==========================================================================
# Stage 3: cleaning (normalise, filter, dedupe) -> feeds the indexer
# ==========================================================================

def _bengali_ratio(text: str) -> float:
    if not text:
        return 0.0
    bengali_chars = sum(1 for ch in text if BENGALI_RANGE[0] <= ord(ch) <= BENGALI_RANGE[1])
    letters = sum(1 for ch in text if ch.isalpha())
    return bengali_chars / letters if letters else 0.0


def _passes_language_filter(text: str) -> bool:
    """Keeps only articles that are predominantly Bengali script. This
    pipeline gathers Bengali facts exclusively, so anything below the
    threshold - stray English/mixed-language pages that occasionally show
    up in these sitemaps - is dropped, regardless of source."""
    return _bengali_ratio(text) >= MIN_BENGALI_RATIO


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _content_hash(title: str, text: str) -> str:
    """Hash of normalised title+text so re-published/mirrored articles under
    different URLs still get deduped, not just exact-URL repeats."""
    key = (title.strip().lower() + "||" + text.strip().lower())
    key = re.sub(r"\s+", " ", key)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def clean_corpus(raw_path: Path = RAW_CORPUS_FILE, clean_path: Path = CLEAN_CORPUS_FILE):
    """Normalises Unicode, drops thin/non-Bengali/duplicate articles, and
    writes the cleaned corpus that the chunker/indexer consumes."""
    if not raw_path.exists():
        log.error(f"Raw corpus not found at {raw_path}. Run the scrape stage first.")
        return

    seen_hashes: set[str] = set()
    kept, dropped_short, dropped_lang, dropped_dupe = 0, 0, 0, 0

    with open(raw_path, "r", encoding="utf-8") as fin, open(clean_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            title = _normalise_text(doc.get("title", ""))
            text = _normalise_text(doc.get("text", ""))

            if len(text) < MIN_BODY_CHARS:
                dropped_short += 1
                continue

            if not _passes_language_filter(text):
                dropped_lang += 1
                continue

            h = _content_hash(title, text)
            if h in seen_hashes:
                dropped_dupe += 1
                continue
            seen_hashes.add(h)

            doc["title"] = title
            doc["text"] = text
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    log.info(
        f"Cleaning complete. Kept {kept} articles. "
        f"Dropped: {dropped_short} too short, {dropped_lang} non-Bengali, {dropped_dupe} duplicates."
    )


# ==========================================================================
# Stage 4: chunking
# ==========================================================================

def load_and_chunk_documents(file_path: Path) -> list[dict]:
    """Loads the cleaned dataset and splits text using Bengali punctuation
    boundaries."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line_str = line.strip()
            if line_str:
                documents.append(json.loads(line_str))

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "।", " ", ""],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )

    chunked_data = []
    for doc in documents:
        splits = text_splitter.split_text(doc.get("text", ""))
        for split in splits:
            chunked_data.append({
                "url": doc.get("url", ""),
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "date": doc.get("date", "Unknown Date"),
                "text": split,
            })

    return chunked_data


# ==========================================================================
# Stage 5: hybrid dense + sparse embedding, checkpointed FAISS index
# ==========================================================================

def generate_hybrid_embeddings_with_checkpoint(
    chunked_data: list,
    model,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    batch_size: int = 32,
):
    """Generates dense vectors and sparse lexical weights with state
    checkpointing, so a killed/interrupted run can resume cleanly.

    Vectors are accumulated to disk here rather than added straight into a
    FAISS index, keeping embedding (this function, GPU-bound, resumable)
    separate from indexing (build_hnsw_index(), a second pass once every
    chunk is embedded) - so an interrupted embedding run never has to
    rebuild a partially-populated index.
    """
    import faiss

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = checkpoint_dir / "dense_vectors.npy"
    metadata_path = checkpoint_dir / "metadata.pkl"
    sparse_path = checkpoint_dir / "sparse_weights.pkl"
    state_path = checkpoint_dir / "checkpoint_state.pkl"

    if vectors_path.exists() and metadata_path.exists() and sparse_path.exists() and state_path.exists():
        log.info("Resuming embedding from existing checkpoint...")
        dense_vecs_all = list(np.load(vectors_path))
        with open(metadata_path, "rb") as f:
            metadata_map = pickle.load(f)
        with open(sparse_path, "rb") as f:
            sparse_map = pickle.load(f)
        with open(state_path, "rb") as f:
            start_index = pickle.load(f)
    else:
        log.info("Initialising new embedding checkpoint...")
        dense_vecs_all, metadata_map, sparse_map, start_index = [], {}, {}, 0

    total_chunks = len(chunked_data)

    for current_index in range(start_index, total_chunks, batch_size):
        batch_chunks = chunked_data[current_index: current_index + batch_size]
        text_batch = [chunk["text"] for chunk in batch_chunks]

        encoded = model.encode(
            text_batch,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size,
            # Chunks are capped at 800 chars (chunk_size in load_and_chunk_
            # documents), so BGE-M3's default 8192-token budget is way more
            # than needed here and just burns VRAM headroom that could
            # otherwise go toward a bigger batch_size.
            max_length=512,
        )

        dense_vecs = np.array(encoded["dense_vecs"], dtype=np.float32)
        faiss.normalize_L2(dense_vecs)  # L2 normalise up front for inner-product cosine similarity
        sparse_weights = encoded["lexical_weights"]

        for offset, chunk in enumerate(batch_chunks):
            vector_id = current_index + offset
            dense_vecs_all.append(dense_vecs[offset])
            metadata_map[vector_id] = chunk
            sparse_map[vector_id] = sparse_weights[offset]

        next_index = current_index + len(batch_chunks)

        np.save(vectors_path, np.stack(dense_vecs_all))
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata_map, f)
        with open(sparse_path, "wb") as f:
            pickle.dump(sparse_map, f)
        with open(state_path, "wb") as f:
            pickle.dump(next_index, f)

        log.info(f"Embedded [{next_index}/{total_chunks}] chunks.")

    return np.stack(dense_vecs_all), metadata_map, sparse_map


def build_hnsw_index(
    dense_vecs: np.ndarray,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
):
    """Builds an IndexHNSWFlat over the full vector set.

    Unlike IndexIVFFlat, HNSW needs no training step - vectors go straight
    into the graph. M controls graph connectivity (build-time, fixed once
    built); ef_construction trades build speed for graph quality; ef_search
    trades query speed for recall and can be changed later on the loaded
    index without rebuilding it.

    dense_vecs must already be L2-normalised (generate_hybrid_embeddings_
    with_checkpoint does this), so inner product == cosine similarity here.
    """
    import faiss

    n_vectors, dim = dense_vecs.shape
    log.info(f"Building IndexHNSWFlat: {n_vectors} vectors, dim={dim}, M={M}")

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    t0 = time.time()
    index.add(dense_vecs)
    log.info(f"Added {index.ntotal} vectors in {round(time.time() - t0, 2)}s")

    index.hnsw.efSearch = ef_search

    index_path = checkpoint_dir / "factcheck_index.faiss"
    faiss.write_index(index, str(index_path))
    log.info(f"Saved HNSW index to {index_path} (M={M}, efSearch={ef_search})")

    return index


def _select_device_and_batch_size() -> tuple[str, int]:
    """Picks the GPU (falling back to CPU) and a batch_size sized to
    actually use the available VRAM instead of leaving it idle.

    Thresholds below are tuned for a 6GB card (e.g. RTX 3050) running
    BGE-M3 in fp16 with max_length=512: batch_size=64 comfortably fits in
    ~5GB+ free, with headroom below that. If you hit a CUDA OOM anyway
    (another process sharing the GPU, a longer max_length, etc.), lower
    batch_size manually or drop these thresholds a notch.
    """
    import torch

    if not torch.cuda.is_available():
        log.warning("No CUDA GPU detected - falling back to CPU. This will be much slower.")
        return "cpu", 8

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    log.info(
        f"CUDA device: {torch.cuda.get_device_name(0)} "
        f"({free_gb:.1f} GB free / {total_bytes / (1024 ** 3):.1f} GB total)"
    )

    if free_gb >= 5:
        batch_size = 64
    elif free_gb >= 3:
        batch_size = 32
    else:
        batch_size = 16

    torch.backends.cudnn.benchmark = True
    return "cuda", batch_size


def run_index_stage():
    if not CLEAN_CORPUS_FILE.exists():
        log.error(f"Clean corpus not found at {CLEAN_CORPUS_FILE}. Run the clean stage first.")
        return

    t0 = time.time()
    chunked_data = load_and_chunk_documents(CLEAN_CORPUS_FILE)
    log.info(f"Chunked {len(chunked_data)} segments in {round(time.time() - t0, 2)}s")

    device, batch_size = _select_device_and_batch_size()
    log.info(f"Loading BGEM3FlagModel on {device} (batch_size={batch_size})...")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=(device == "cuda"), device=device)

    t0 = time.time()
    dense_vecs, metadata_map, sparse_map = generate_hybrid_embeddings_with_checkpoint(
        chunked_data=chunked_data, model=model, batch_size=batch_size
    )
    log.info(f"Embedding completed in {round(time.time() - t0, 2)}s. Total vectors: {len(dense_vecs)}.")

    t0 = time.time()
    index = build_hnsw_index(dense_vecs)
    log.info(
        f"HNSW indexing completed in {round(time.time() - t0, 2)}s. "
        f"Total vectors: {index.ntotal}, Total sparse entries: {len(sparse_map)}."
    )


# ==========================================================================
# Orchestration
# ==========================================================================

STAGES = ["discover", "scrape", "clean", "chunk_index"]


def main():
    parser = argparse.ArgumentParser(description="Scrape + build hybrid FAISS index, all in one directory.")
    parser.add_argument("--skip-scrape", action="store_true", help="Reuse existing raw corpus, skip discovery/scraping.")
    parser.add_argument("--only", choices=STAGES, help="Run only a single stage.")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent scraper threads (default {DEFAULT_MAX_WORKERS}). Lower this if a source starts 429ing.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.only:
        if args.only in ("discover", "scrape"):
            run_scrape_stage(max_workers=args.workers)
        elif args.only == "clean":
            clean_corpus()
        elif args.only == "chunk_index":
            run_index_stage()
        return

    if not args.skip_scrape:
        run_scrape_stage(max_workers=args.workers)
    else:
        log.info("Skipping scrape stage (--skip-scrape).")

    clean_corpus()
    run_index_stage()


if __name__ == "__main__":
    main()