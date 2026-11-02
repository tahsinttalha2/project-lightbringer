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
            dense_vectors.f32              (preallocated memmap, written per batch slice)
            sparse_weights.stream          (append-only, one pickle record per vector)
            checkpoint_state.pkl           (start_index + stream offset + corpus fingerprint)
            factcheck_index.faiss          (IndexHNSWFlat, written once at the end)
            metadata.pkl                   (consolidated once embedding finishes)
            sparse_weights.pkl             (consolidated once embedding finishes)

Run:
    python pipeline.py                     # full pipeline
    python pipeline.py --skip-scrape       # reuse existing raw corpus, just clean+index
    python pipeline.py --only clean        # run a single stage
    python pipeline.py --workers 16        # more concurrent scraper threads
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
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
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Sends log records through tqdm.write() instead of straight to
    stderr. A normal handler and an active tqdm bar fight over the same
    line (tqdm redraws via \\r) - every log.info/warning call would tear
    the bar in half and leave fragments on screen. tqdm.write() pauses the
    bar, prints the line cleanly above it, then redraws."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[TqdmLoggingHandler()],
)
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
    "rumorscanner": "https://rumorscanner.com/sitemap.xml",
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
# Characters inspected when deciding an article's language. Well above
# MIN_BODY_CHARS, so every article that survives the length filter still gets
# a substantial sample.
LANG_SAMPLE_CHARS = 2000

RETRY_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0

# Scraping is I/O-bound (waiting on remote servers), so a thread pool gives
# a near-linear speedup over the old one-URL-at-a-time loop. If a particular
# site starts returning 429s, drop --workers for that run rather than
# raising it further — the per-URL backoff above still applies per thread.
DEFAULT_MAX_WORKERS = 8

# How often the corpus/completed/failed handles are flushed during scraping.
# Anything buffered when a crash happens is just re-scraped next run, so this
# trades a few seconds of redone work for not paying an fsync per URL.
SCRAPE_FLUSH_SECONDS = 2.0

# Chunking parameters live here (rather than inline in load_and_chunk_documents)
# because the embedding checkpoint fingerprints them: changing either value
# changes the chunk boundaries, which must invalidate an in-progress resume.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

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
                # One call, one lxml parse. The previous version called
                # trafilatura.extract() and trafilatura.extract_metadata()
                # separately, which parsed the same HTML document twice and
                # roughly doubled the CPU cost of this stage. Asking for
                # JSON output with metadata gets both out of a single pass.
                payload = trafilatura.extract(
                    res.text,
                    include_comments=False,
                    include_tables=False,
                    with_metadata=True,
                    output_format="json",
                )
                if not payload:
                    return "permanent_fail", None

                try:
                    parsed = json.loads(payload)
                except (ValueError, TypeError):
                    return "permanent_fail", None

                title_text = (parsed.get("title") or "").strip()
                date_text = (parsed.get("date") or "").strip() or "Unknown Date"
                body_text = (parsed.get("text") or "").strip()

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

    # .iterrows() builds a fresh Series per row and is pathologically slow
    # on a 100k+ row catalogue. Zipping the raw numpy columns does the same
    # job in a fraction of the time.
    todo = [
        (url, source)
        for url, source in zip(df_targets["url"].to_numpy(), df_targets["source"].to_numpy())
        if url not in completed_urls and url not in failed_urls
    ]

    log.info(
        f"Starting extraction loop. {len(completed_urls)} already processed, "
        f"{len(failed_urls)} permanently failed, {len(todo)} remaining. "
        f"Using {max_workers} concurrent workers."
    )

    write_lock = threading.Lock()
    last_source = None
    last_flush = time.monotonic()

    # The log files are opened once for the whole run instead of being
    # reopened per URL (the old log_url() did open+write+close while holding
    # the write lock, i.e. two syscalls' worth of latency serialised across
    # every worker). Flushes are batched on a timer rather than fired per
    # record; SCRAPE_FLUSH_SECONDS bounds how much is at risk on a crash,
    # and anything unflushed is simply re-scraped on the next run because
    # the completed log is what defines "already done".
    with open(RAW_CORPUS_FILE, "a", encoding="utf-8") as corpus, \
            open(COMPLETED_LOG, "a", encoding="utf-8") as completed_fh, \
            open(FAILED_LOG, "a", encoding="utf-8") as failed_fh, \
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool, \
            tqdm(total=len(todo), desc="Scraping", unit="url") as pbar:

        def _flush_all():
            corpus.flush()
            completed_fh.flush()
            failed_fh.flush()

        future_to_target = {
            pool.submit(scrape_article, url, source): (url, source)
            for url, source in todo
        }

        for future in concurrent.futures.as_completed(future_to_target):
            url, source = future_to_target[future]
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
                    completed_fh.write(f"{url}\n")
                elif status == "permanent_fail":
                    failed_fh.write(f"{url}\n")
                else:
                    log.info(f"Transient failure on {url}, will retry on next run.")

                now = time.monotonic()
                if now - last_flush >= SCRAPE_FLUSH_SECONDS:
                    _flush_all()
                    last_flush = now

            pbar.update(1)
            # Updating the postfix redraws the whole bar, so only do it when
            # the source actually changes rather than on every single URL.
            if source != last_source:
                pbar.set_postfix_str(source)
                last_source = source

        with write_lock:
            _flush_all()

    log.info("Scraping stage finished.")


# ==========================================================================
# Stage 3: cleaning (normalise, filter, dedupe) -> feeds the indexer
# ==========================================================================

def _bengali_ratio(text: str) -> float:
    """Fraction of alphabetic characters that fall in the Bengali block.

    Only the first LANG_SAMPLE_CHARS characters are inspected. Language is a
    whole-document property here - an article is Bengali or it isn't - so a
    couple of thousand characters settle it just as decisively as scanning a
    20KB body, and this used to be two full Python-level passes over every
    character of every article in the corpus.
    """
    if not text:
        return 0.0
    sample = text[:LANG_SAMPLE_CHARS]
    lo, hi = BENGALI_RANGE
    bengali_chars = 0
    letters = 0
    for ch in sample:
        if ch.isalpha():
            letters += 1
            if lo <= ord(ch) <= hi:
                bengali_chars += 1
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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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

def _atomic_pickle_dump(obj, path: Path):
    """Writes a pickle to `path` atomically: dump to a tmp file in the same
    directory, flush+fsync it to disk, then os.replace() it onto the real
    path. os.replace is an atomic rename on POSIX and Windows, so a kill
    mid-write leaves either the old complete file or the new complete file
    - never a truncated one that raises EOFError on load."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)



# Bumped whenever the on-disk checkpoint layout changes in a way that makes
# older checkpoints unreadable. v1 stored metadata/sparse as whole-dict
# pickles rewritten on every checkpoint; v2 streams sparse weights and drops
# the metadata checkpoint entirely (see below). A v1 checkpoint is detected
# and discarded rather than misread.
CHECKPOINT_FORMAT_VERSION = 2


def _checkpoint_fingerprint(chunked_data: list, embedding_dim: int) -> str:
    """Identifies the exact chunk set a checkpoint belongs to.

    Resuming is only valid if chunk i means the same thing this run as it did
    last run, because start_index addresses chunks positionally. If the corpus
    is re-scraped or re-cleaned, or the chunk parameters change, the chunk
    sequence shifts and a stale start_index silently pairs vectors with the
    wrong text - no exception, just a quietly wrong index.

    np.memmap does not catch this on its own: mode="r+" raises when the file
    is too small for the requested shape, but silently maps a prefix when the
    file is too large, so a shrunken corpus would resume without complaint.

    Hashing the total count, the geometry, the chunk parameters and a spread
    of probe chunks costs microseconds and turns that silent corruption into
    a clean "start over".
    """
    h = hashlib.sha256()
    h.update(
        f"v{CHECKPOINT_FORMAT_VERSION}|{len(chunked_data)}|{embedding_dim}|"
        f"{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode("utf-8")
    )

    n = len(chunked_data)
    probes = {0, 1, 2, n // 3, n // 2, (2 * n) // 3, n - 3, n - 2, n - 1}
    for i in sorted(p for p in probes if 0 <= p < n):
        chunk = chunked_data[i]
        h.update(f"|{i}|".encode("utf-8"))
        h.update(chunk.get("url", "").encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(chunk.get("text", "").encode("utf-8", "replace"))

    return h.hexdigest()


def _encode_with_oom_backoff(model, text_batch: list[str], max_length: int = 512):
    """Encodes one batch, halving it and retrying on CUDA OOM.

    A 6GB card is close enough to the edge that an unlucky batch of long
    chunks can OOM even when the configured batch size normally fits. Without
    this, that kills the process and the next run just walks back into the
    same batch. Returns (encoded, n_texts_actually_encoded) so the caller can
    consume only what was encoded and lower its batch size accordingly.

    The batch_size argument passed down to FlagEmbedding is deliberately one
    larger than the slice. encode_single_device() spins up two internal tqdm
    bars ("pre tokenize" / "Inference Embeddings") per call, skipping them
    only when len(sentences) < batch_size; passing len+1 flips that check to
    True and mutes them. The slice is processed in a single internal pass
    either way, so nothing about the actual encoding changes.

    max_length=512 rather than BGE-M3's 8192 default: chunks are capped at
    CHUNK_SIZE characters, so the larger budget only burns VRAM headroom that
    could otherwise go toward a bigger batch.
    """
    import torch

    texts = text_batch
    while True:
        try:
            encoded = model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=len(texts) + 1,
                max_length=max_length,
            )
            return encoded, len(texts)
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            texts = texts[: max(1, len(texts) // 2)]
            log.warning(f"CUDA OOM on this batch; retrying with {len(texts)} texts.")


def _read_sparse_stream(path: Path, expected: int) -> list[dict]:
    """Replays the append-only sparse stream back into a list of per-vector
    lexical weight dicts."""
    weights: list[dict] = []
    with open(path, "rb") as f:
        while len(weights) < expected:
            try:
                weights.append(pickle.load(f))
            except EOFError:
                break
    return weights


def generate_hybrid_embeddings_with_checkpoint(
    chunked_data: list,
    model,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    batch_size: int = 32,
    checkpoint_every_n_batches: int = 20,
    checkpoint_every_seconds: float = 30.0,
):
    """Generates dense vectors and sparse lexical weights with state
    checkpointing, so a killed/interrupted run can resume cleanly.

    Vectors are accumulated to disk here rather than added straight into a
    FAISS index, keeping embedding (this function, GPU-bound, resumable)
    separate from indexing (build_hnsw_index(), a second pass once every
    chunk is embedded) - so an interrupted embedding run never has to
    rebuild a partially-populated index.

    Per-batch cost is constant, independent of how far into the run we are.
    Three separate O(n^2)-over-a-run behaviours were removed to get there:

    - Dense vectors used to be re-stacked and rewritten in full every batch.
      They now go into a preallocated memmap at the batch's exact slice.
    - Sparse weights used to be held in a dict and re-pickled in full at every
      checkpoint, so late checkpoints were serialising hundreds of MB. They
      are now appended to a stream, one pickle record per vector, and only
      assembled into a dict once at the end. They are never held in RAM
      during the loop at all.
    - Metadata used to be checkpointed the same way. It no longer needs
      checkpointing in any form: metadata_map[i] is just chunked_data[i],
      which the fingerprint now guarantees is identical across resumes, so it
      is rebuilt for free at the end instead of being written 1000 times.

    The memmap flush also moved onto the checkpoint interval. flush() msyncs
    the whole mapping, and dense data ahead of the last checkpointed
    start_index is re-embedded on resume regardless, so flushing it more often
    than the checkpoint bought nothing.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    import faiss

    # BGE-M3's dense output is a fixed 1024-dim vector per input.
    EMBEDDING_DIM = 1024

    vectors_path = checkpoint_dir / "dense_vectors.f32"
    sparse_stream_path = checkpoint_dir / "sparse_weights.stream"
    state_path = checkpoint_dir / "checkpoint_state.pkl"
    # Final consolidated artefacts, written once when embedding completes.
    # Format unchanged from before, so anything downstream that loads these
    # (the retrieval side) needs no changes.
    metadata_path = checkpoint_dir / "metadata.pkl"
    sparse_path = checkpoint_dir / "sparse_weights.pkl"

    total_chunks = len(chunked_data)
    fingerprint = _checkpoint_fingerprint(chunked_data, EMBEDDING_DIM)
    expected_bytes = total_chunks * EMBEDDING_DIM * 4

    start_index = 0
    sparse_offset = 0
    dense_memmap = None
    resumed = False

    if vectors_path.exists() and sparse_stream_path.exists() and state_path.exists():
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)

            if not isinstance(state, dict):
                raise ValueError("pre-v2 checkpoint (bare start_index)")
            if state.get("version") != CHECKPOINT_FORMAT_VERSION:
                raise ValueError(f"checkpoint format v{state.get('version')} is not readable")
            if state.get("fingerprint") != fingerprint:
                raise ValueError("chunk set has changed since this checkpoint was written")

            actual_bytes = vectors_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"dense vector file is {actual_bytes} bytes, expected {expected_bytes}"
                )

            start_index = int(state["start_index"])
            sparse_offset = int(state["sparse_offset"])
            if not 0 <= start_index <= total_chunks:
                raise ValueError(f"start_index {start_index} outside 0..{total_chunks}")
            if sparse_offset > sparse_stream_path.stat().st_size:
                raise ValueError("sparse stream is shorter than the checkpointed offset")

            # mode="r+": open the existing file in place, writable, without
            # touching what's already on disk - resuming just continues
            # writing into the same slices future batches need.
            dense_memmap = np.memmap(
                vectors_path, dtype=np.float32, mode="r+", shape=(total_chunks, EMBEDDING_DIM)
            )
            resumed = True
            log.info(
                f"Resuming embedding from checkpoint at chunk {start_index}/{total_chunks}."
            )
        except (EOFError, pickle.UnpicklingError, OSError, ValueError, KeyError, TypeError) as e:
            # Either a genuinely corrupt checkpoint, or - far more likely now
            # that writes are atomic - a checkpoint that belongs to a
            # different corpus. Neither is safe to resume from, and neither
            # should crash the run: start clean.
            log.error(
                f"Cannot resume checkpoint at {checkpoint_dir} ({e}); starting a fresh one."
            )
            start_index, sparse_offset, dense_memmap, resumed = 0, 0, None, False

    if not resumed:
        log.info("Initialising new embedding checkpoint...")
        # mode="w+": create (or overwrite) the file at its full final size
        # up front, zero-filled. Every batch below writes into its own
        # slice of this same file - never a full rewrite.
        dense_memmap = np.memmap(
            vectors_path, dtype=np.float32, mode="w+", shape=(total_chunks, EMBEDDING_DIM)
        )

    last_checkpoint_time = time.monotonic()
    batches_since_checkpoint = 0

    # "r+b" + seek + truncate on resume: a crash between the last checkpoint
    # and the kill may have left extra complete records past sparse_offset
    # (they belong to batches that will now be redone). Dropping them keeps
    # record N in the stream aligned with vector N in the memmap, which is
    # the whole basis for reading the stream back positionally.
    stream_mode = "r+b" if resumed else "wb"
    with open(sparse_stream_path, stream_mode) as sparse_fh:
        if resumed:
            sparse_fh.seek(sparse_offset)
            sparse_fh.truncate()

        current_index = start_index
        with tqdm(total=total_chunks, initial=start_index, desc="Embedding", unit="chunk") as pbar:
            while current_index < total_chunks:
                batch_chunks = chunked_data[current_index: current_index + batch_size]
                text_batch = [chunk["text"] for chunk in batch_chunks]

                encoded, used_batch_size = _encode_with_oom_backoff(model, text_batch)

                if used_batch_size < len(text_batch):
                    # The batch was too big for the card and the helper fell
                    # back to a smaller slice. Shrink the working batch size
                    # so the rest of the run stops trying the size that just
                    # OOM'd, and only consume what was actually encoded.
                    batch_chunks = batch_chunks[:used_batch_size]
                    batch_size = used_batch_size
                    log.warning(f"Reducing embedding batch_size to {batch_size} for the rest of the run.")

                dense_vecs = np.asarray(encoded["dense_vecs"], dtype=np.float32)
                faiss.normalize_L2(dense_vecs)  # L2 normalise up front for inner-product cosine similarity

                next_index = current_index + len(batch_chunks)

                # Write straight into this batch's slice - cost is always just
                # this batch, never the whole array so far.
                dense_memmap[current_index:next_index] = dense_vecs

                # Append-only, one record per vector, in vector order. Never
                # rewritten, never accumulated in memory.
                for weights in encoded["lexical_weights"]:
                    pickle.dump(weights, sparse_fh, protocol=pickle.HIGHEST_PROTOCOL)

                batches_since_checkpoint += 1
                elapsed_since_checkpoint = time.monotonic() - last_checkpoint_time
                should_checkpoint = (
                    batches_since_checkpoint >= checkpoint_every_n_batches
                    or elapsed_since_checkpoint >= checkpoint_every_seconds
                    or next_index >= total_chunks  # always checkpoint the final batch
                )

                if should_checkpoint:
                    # Order matters for crash-safety: state_path is written
                    # LAST, and only after the data it describes is durable.
                    # If the process dies partway through this block, the
                    # on-disk state still points at the previous checkpoint,
                    # so a resume redoes at most checkpoint_every_n_batches
                    # batches / checkpoint_every_seconds worth of work. It
                    # never resumes past data that wasn't actually saved.
                    dense_memmap.flush()
                    sparse_fh.flush()
                    os.fsync(sparse_fh.fileno())
                    _atomic_pickle_dump(
                        {
                            "version": CHECKPOINT_FORMAT_VERSION,
                            "start_index": next_index,
                            "sparse_offset": sparse_fh.tell(),
                            "fingerprint": fingerprint,
                        },
                        state_path,
                    )
                    last_checkpoint_time = time.monotonic()
                    batches_since_checkpoint = 0

                pbar.update(len(batch_chunks))
                current_index = next_index

    dense_memmap.flush()

    # Metadata is pure function of chunked_data, so it costs one pass here
    # rather than a rewrite per checkpoint. Values are references into
    # chunked_data, not copies.
    metadata_map = {i: chunk for i, chunk in enumerate(chunked_data)}

    sparse_list = _read_sparse_stream(sparse_stream_path, total_chunks)
    if len(sparse_list) != total_chunks:
        log.warning(
            f"Sparse stream holds {len(sparse_list)} records for {total_chunks} chunks; "
            "the checkpoint may be incomplete."
        )
    sparse_map = {i: weights for i, weights in enumerate(sparse_list)}
    del sparse_list

    _atomic_pickle_dump(metadata_map, metadata_path)
    _atomic_pickle_dump(sparse_map, sparse_path)
    log.info(f"Wrote consolidated {metadata_path.name} and {sparse_path.name}.")

    # The memmap is returned as-is rather than copied into RAM. It is already
    # an ndarray subclass, and build_hnsw_index() reads it in slices, so the
    # OS pages it in and out instead of us holding an extra ~800MB resident
    # next to the index being built.
    return dense_memmap, metadata_map, sparse_map


def build_hnsw_index(
    dense_vecs: np.ndarray,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    add_chunk_size: int = 50_000,
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
    # dense_vecs is typically a memmap. Adding it in slices means FAISS only
    # ever needs one slice materialised at a time (np.ascontiguousarray on a
    # memmap slice copies just that slice), so peak RAM here is the index
    # itself plus ~200MB rather than the index plus the entire vector set.
    # Slicing also gives the progress bar something to report on what is
    # otherwise a long silent call.
    with tqdm(total=n_vectors, desc="Indexing", unit="vec") as pbar:
        for begin in range(0, n_vectors, add_chunk_size):
            end = min(begin + add_chunk_size, n_vectors)
            index.add(np.ascontiguousarray(dense_vecs[begin:end], dtype=np.float32))
            pbar.update(end - begin)
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
    BGE-M3 in fp16. max_length=512 is only a ceiling - transformers pad to
    the longest sequence in the batch, and CHUNK_SIZE-character Bengali
    chunks tokenise to roughly 250-350 tokens, so real batches sit well under
    that ceiling and leave headroom the old batch_size=64 never used.

    An OOM is no longer fatal: _encode_with_oom_backoff() halves the batch
    and the embedding loop keeps the lower size for the rest of the run. Set
    BGE_BATCH_SIZE to pin a specific value and skip this heuristic entirely.
    """
    import torch

    override = os.environ.get("BGE_BATCH_SIZE")

    if not torch.cuda.is_available():
        log.warning("No CUDA GPU detected - falling back to CPU. This will be much slower.")
        return "cpu", int(override) if override else 8

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    log.info(
        f"CUDA device: {torch.cuda.get_device_name(0)} "
        f"({free_gb:.1f} GB free / {total_bytes / (1024 ** 3):.1f} GB total)"
    )

    if override:
        batch_size = int(override)
        log.info(f"Using BGE_BATCH_SIZE={batch_size} from the environment.")
    elif free_gb >= 5:
        batch_size = 96
    elif free_gb >= 3:
        batch_size = 48
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

    # Release BGE-M3 before the index build. Nothing below this point touches
    # the model, and holding ~1GB of weights (plus the CUDA context and
    # allocator cache) while FAISS builds a graph over the full vector set is
    # the tightest moment for RAM on a 16GB machine.
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

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