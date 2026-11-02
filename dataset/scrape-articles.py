#!/usr/bin/env python3
"""
scrape-articles.py -- Phase 1 of the LightBringer crawl: download and save raw pages.

What it does
    1. Reads sitemaps (including sitemap indexes) and builds a list of page URLs.
    2. Checks each site's robots.txt and skips URLs it disallows.
    3. Downloads each page politely and saves the raw HTML (gzip-compressed).
    4. Writes one small .json "receipt" per URL, then rebuilds manifest.csv from them.

What it does NOT do
    It does not parse articles, find YouTube links or check videos. Those are
    Phase 2 and Phase 3, and they work offline from the files saved here.

Resume rule (the only state that matters)
    A URL is "done" when its receipt (.json) exists AND records a final outcome
    ("saved", "http_error" such as 404, or "blocked_by_robots"). If you stop the
    script at any moment, just run the same command again: finished URLs are
    skipped, and temporary failures ("failed": timeouts, 429, 5xx) are retried.
    There are no separate key/progress files that could drift out of sync, and
    manifest.csv is rebuilt from the receipts, so it can never disagree with them.

Output layout (inside --out-dir, default "data")
    html/<key>.html.gz     raw page exactly as received, compressed
    meta/<key>.json        receipt: url, status, time fetched, filename, error ...
    sitemaps/<key>.xml     the sitemap files you crawled from (provenance)
    robots/<host>__<date>.txt   robots.txt as it was when you crawled
    url_list.csv           every URL found in the sitemaps
    manifest.csv           one row per URL attempted (rebuilt every run)
    crawl.log              what happened, with timestamps

Safety default
    Without --all the script stops after 20 NEW downloads per run. Use that for
    testing. Only pass --all once you have decided you are allowed to crawl.

Install:  pip install requests
Example:  python scrape-articles.py --sources rumour-scanner jachai \
              --contact-email you@example.com --limit 20

Sources
    The sites are hardcoded in SOURCES below. Name the ones you want with --sources;
    nothing is crawled by default. For each source the script finds the sitemap
    from the Sitemap: lines in that site's robots.txt, and falls back to the usual
    sitemap addresses if there are none.
"""

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET

import requests

REQUEST_TIMEOUT = 20      # seconds to wait for a server before giving up
MAX_RETRIES = 3           # retries for timeouts / 429 / 5xx
BACKOFF_BASE = 2.0        # wait = BACKOFF_BASE * 2**attempt (+ small random extra)
SITEMAP_PAUSE = 1.0       # seconds between sitemap fetches

# The fact-checking sites this project collects from. The key is the name used by
# --sources and stored in every receipt and in manifest.csv.
SOURCES = {
    "rumour-scanner": "https://rumorscanner.com/",
    "jachai": "https://jachai.org/",
    "boom-bd": "https://www.boombd.com/",
    "afp-fact-check-bangla": "https://factcheckbangla.afp.com/",
    "dismislab": "https://dismislab.com/",
    "bangla-fact": "https://banglafact.com/",
}

# Tried only when a site's robots.txt lists no Sitemap: line.
FALLBACK_SITEMAP_PATHS = ["/sitemap_index.xml", "/sitemap.xml", "/wp-sitemap.xml"]

# Sources whose own terms or robots.txt need care before a full crawl.
SOURCE_WARNINGS = {
    "afp-fact-check-bangla": (
        "AFP's terms allow only short extracts for research, and its robots.txt blocks AI "
        "crawlers. Get written permission before crawling it in bulk."
    ),
}

# Outcomes that mean "do not try this URL again on resume".
TERMINAL_OUTCOMES = {"saved", "http_error", "blocked_by_robots"}

MANIFEST_FIELDS = [
    "url", "source", "source_host", "sitemap_lastmod", "outcome", "http_status",
    "final_url", "fetched_at_utc", "content_type", "bytes", "html_file", "error",
]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def url_key(url):
    """Stable filename for a URL: same URL -> same name, different URLs -> different names."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def write_atomic(path, data):
    """
    Write bytes to a temp file, then rename it into place. A rename is all-or-nothing,
    so stopping the script mid-write can never leave a half-written file that looks finished.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def setup_logging(log_path):
    log = logging.getLogger("crawler")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(file_handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(console)
    return log


# ----------------------------------------------------------------------
# Fetching with retries
# ----------------------------------------------------------------------

def fetch(url, session, log):
    """
    GET a URL. Returns (response, error_text).

    - Timeouts, connection errors, 429 and 5xx are retried with growing waits.
    - Any other status (200, 404, 403 ...) is returned straight away; the caller decides.
    - If retries run out, the last response (maybe None) is returned with an error text.
    """
    resp = None
    error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as e:
            resp = None
            error = f"{type(e).__name__}: {e}"
        except requests.RequestException as e:
            # Redirect loops, malformed URLs, SSL problems...: retrying will not help.
            return None, f"{type(e).__name__}: {e}"
        else:
            if resp.status_code != 429 and resp.status_code < 500:
                return resp, ""
            error = f"HTTP {resp.status_code}"

        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5)
            if resp is not None:
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = min(max(wait, int(retry_after)), 60)
            log.info(f"{error} on {url} (attempt {attempt + 1}/{MAX_RETRIES}); retrying in {wait:.1f}s")
            time.sleep(wait)

    return resp, f"gave up after {MAX_RETRIES} retries: {error}"


# ----------------------------------------------------------------------
# robots.txt
# ----------------------------------------------------------------------

class RobotsCache:
    """
    Downloads each site's robots.txt once, saves a copy, and answers "may I fetch this URL?".

    Python's built-in parser only does simple prefix matching. It does not understand
    the * and $ wildcards some sites use in their rules. So treat this as a safety net,
    and still read each site's robots.txt yourself.

    If robots.txt cannot be read (server error, timeout), we assume NOT allowed.
    """

    def __init__(self, session, log, robots_dir, user_agent):
        self.session = session
        self.log = log
        self.robots_dir = robots_dir
        self.user_agent = user_agent
        self.parsers = {}

    def allowed(self, url):
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.parsers:
            self.parsers[origin] = self._load(origin, parts.netloc)
        parser = self.parsers[origin]
        if parser is None:
            return False
        return parser.can_fetch(self.user_agent, url)

    def sitemaps_for(self, base_url):
        """Sitemap URLs the site lists in its robots.txt (empty list if none or unreadable)."""
        parts = urlparse(base_url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.parsers:
            self.parsers[origin] = self._load(origin, parts.netloc)
        parser = self.parsers[origin]
        if parser is None:
            return []
        return list(parser.site_maps() or [])

    def _load(self, origin, host):
        robots_url = origin + "/robots.txt"
        resp, error = fetch(robots_url, self.session, self.log)
        if resp is None:
            self.log.warning(f"Could not read {robots_url} ({error}); treating site as disallowed")
            return None

        parser = RobotFileParser()
        if resp.status_code == 200:
            text = resp.content.decode("utf-8", errors="replace")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            safe_host = host.replace(":", "_")
            (self.robots_dir / f"{safe_host}__{stamp}.txt").write_text(text, encoding="utf-8")
            parser.parse(text.splitlines())
            self.log.info(f"Loaded robots.txt for {host}")
        elif resp.status_code == 404:
            parser.allow_all = True
            self.log.info(f"No robots.txt at {host} (404): nothing is disallowed")
        elif resp.status_code in (401, 403):
            parser.disallow_all = True
            self.log.warning(f"robots.txt at {host} returned {resp.status_code}: treating site as disallowed")
        else:
            self.log.warning(f"robots.txt at {host} returned {resp.status_code}; treating site as disallowed")
            return None
        parser.modified()  # marks the rules as loaded; can_fetch() refuses otherwise
        return parser


# ----------------------------------------------------------------------
# Sitemaps
# ----------------------------------------------------------------------

def local_name(tag):
    """'{namespace}loc' -> 'loc'"""
    return tag.rsplit("}", 1)[-1].lower()


def read_sitemap(url, session, log, sitemap_dir, seen, sub_filter):
    """
    Resolve a sitemap URL into a list of (page_url, lastmod) pairs.
    Handles sitemap indexes (sitemaps that list other sitemaps) to any depth.
    sub_filter: if given, inside an index only follow sub-sitemaps whose URL contains it.
    """
    if url in seen:
        return []
    seen.add(url)

    time.sleep(SITEMAP_PAUSE)
    resp, error = fetch(url, session, log)
    if resp is None or resp.status_code != 200:
        status = resp.status_code if resp is not None else error
        log.warning(f"Could not read sitemap {url} ({status})")
        return []

    data = resp.content
    if url.lower().endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except OSError:
            pass  # server already decompressed it
    write_atomic(sitemap_dir / f"{url_key(url)}.xml", data)

    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        log.warning(f"Sitemap is not valid XML: {url} ({e})")
        return []

    entries = []
    for node in root:
        if local_name(node.tag) not in ("sitemap", "url"):
            continue
        loc = ""
        lastmod = ""
        for child in node:
            name = local_name(child.tag)
            if name == "loc" and child.text:
                loc = child.text.strip()
            elif name == "lastmod" and child.text:
                lastmod = child.text.strip()
        if loc:
            entries.append((loc, lastmod))

    kind = local_name(root.tag)
    if kind == "sitemapindex":
        log.info(f"{url} is an index with {len(entries)} sub-sitemaps")
        pages = []
        for sub_url, _ in entries:
            if sub_filter and sub_filter not in sub_url:
                log.info(f"  skipping sub-sitemap (does not contain '{sub_filter}'): {sub_url}")
                continue
            pages.extend(read_sitemap(sub_url, session, log, sitemap_dir, seen, sub_filter))
        return pages
    if kind == "urlset":
        log.info(f"{url} lists {len(entries)} URLs")
        return entries

    log.warning(f"Unrecognised sitemap type '{root.tag}' in {url}")
    return []


# ----------------------------------------------------------------------
# Receipts and manifest
# ----------------------------------------------------------------------

def is_done(meta_path, html_dir):
    """True if this URL has a receipt with a final outcome (and its file, if it claims one)."""
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if meta.get("outcome") not in TERMINAL_OUTCOMES:
        return False
    if meta["outcome"] == "saved" and not (html_dir / meta.get("html_file", "")).exists():
        return False
    return True


def write_receipt(meta_path, record):
    write_atomic(meta_path, json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8"))


def rebuild_manifest(meta_dir, manifest_path):
    """Rebuild manifest.csv from all receipts. Returns a Counter of outcomes."""
    rows = []
    for path in sorted(meta_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, manifest_path)
    return Counter(row.get("outcome", "") for row in rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def interleave(entries):
    """Round-robin across sources, so a limited test run samples every source."""
    groups = {}
    for entry in entries:
        groups.setdefault(entry[2], []).append(entry)
    queues = list(groups.values())
    merged = []
    for i in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if i < len(q):
                merged.append(q[i])
    return merged


def parse_args():
    p = argparse.ArgumentParser(description="Download raw pages from sitemaps, resumably and politely.")
    p.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=[],
                   help="Which hardcoded sources to crawl. Nothing is crawled unless you name it.")
    p.add_argument("--sitemaps", nargs="+", default=[],
                   help="Extra sitemap URLs to crawl as well (an index or a plain sitemap).")
    p.add_argument("--contact-email", required=True,
                   help="Your real email. It goes into the User-Agent so site owners can reach you.")
    p.add_argument("--out-dir", default="data", help="Folder for everything the crawl produces.")
    p.add_argument("--limit", type=int, default=20,
                   help="Max NEW pages to download this run (default 20). Already-done URLs don't count.")
    p.add_argument("--all", action="store_true",
                   help="Remove the limit and download everything not yet done.")
    p.add_argument("--sub-sitemap-contains", default="",
                   help="Inside a sitemap index, only follow sub-sitemaps whose URL contains this text "
                        "(for example 'post-sitemap' to skip category and tag pages).")
    p.add_argument("--delay-min", type=float, default=1.5, help="Minimum pause between page fetches (s).")
    p.add_argument("--delay-max", type=float, default=3.0, help="Maximum pause between page fetches (s).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.delay_max < args.delay_min:
        sys.exit("--delay-max must be >= --delay-min")
    if not args.sources and not args.sitemaps:
        sys.exit("Nothing to crawl. Pass --sources (choices: " + ", ".join(sorted(SOURCES)) + ") or --sitemaps.")

    out = Path(args.out_dir)
    html_dir = out / "html"
    meta_dir = out / "meta"
    sitemap_dir = out / "sitemaps"
    robots_dir = out / "robots"
    for d in (html_dir, meta_dir, sitemap_dir, robots_dir):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_logging(out / "crawl.log")
    user_agent = f"LightBringerResearchCrawler/1.0 (academic research; contact: {args.contact_email})"
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    limit = None if args.all else args.limit
    log.info(f"Starting. limit={'none' if limit is None else limit}, out-dir={out}")

    robots = RobotsCache(session, log, robots_dir, user_agent)

    # 1. Work out which sitemaps to read for each chosen source.
    jobs = []  # (source name, sitemap URL)
    for name in args.sources:
        if name in SOURCE_WARNINGS:
            log.warning(f"[{name}] {SOURCE_WARNINGS[name]}")
        base = SOURCES[name]
        sitemap_urls = robots.sitemaps_for(base)
        if sitemap_urls:
            log.info(f"[{name}] robots.txt lists {len(sitemap_urls)} sitemap(s)")
        else:
            sitemap_urls = [base.rstrip("/") + path for path in FALLBACK_SITEMAP_PATHS]
            log.info(f"[{name}] no Sitemap: line in robots.txt; trying the usual addresses")
        jobs.extend((name, u) for u in sitemap_urls)
    jobs.extend(("custom", u) for u in args.sitemaps)

    # 2. Build the URL list from the sitemaps.
    found = {}  # url -> (lastmod, source); the same URL in two sitemaps counts once
    seen_sitemaps = set()
    for name, sm_url in jobs:
        for loc, lastmod in read_sitemap(sm_url, session, log, sitemap_dir, seen_sitemaps,
                                         args.sub_sitemap_contains):
            found.setdefault(loc, (lastmod, name))
    entries = interleave([(url, lastmod, name) for url, (lastmod, name) in found.items()])
    per_source = Counter(e[2] for e in entries)
    log.info(f"{len(entries)} unique URLs found: " + ", ".join(f"{k}={v}" for k, v in sorted(per_source.items())))
    with open(out / "url_list.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "source", "sitemap_lastmod"])
        writer.writerows((url, name, lastmod) for url, lastmod, name in entries)

    # 3. Download.
    attempted = 0
    skipped_done = 0
    limit_hit = False

    try:
        for index, (url, lastmod, source) in enumerate(entries, start=1):
            key = url_key(url)
            meta_path = meta_dir / f"{key}.json"

            if is_done(meta_path, html_dir):
                skipped_done += 1
                continue
            if limit is not None and attempted >= limit:
                limit_hit = True
                break
            attempted += 1

            record = {
                "url": url,
                "source": source,
                "source_host": urlparse(url).netloc,
                "sitemap_lastmod": lastmod,
                "outcome": "",
                "http_status": "",
                "final_url": "",
                "fetched_at_utc": now_utc(),
                "content_type": "",
                "bytes": "",
                "html_file": "",
                "error": "",
            }

            if not robots.allowed(url):
                record["outcome"] = "blocked_by_robots"
                write_receipt(meta_path, record)
                log.info(f"[{index}/{len(entries)}] blocked by robots.txt: {url}")
                continue

            time.sleep(random.uniform(args.delay_min, args.delay_max))
            resp, error = fetch(url, session, log)
            record["fetched_at_utc"] = now_utc()

            if resp is None:
                record["outcome"] = "failed"
                record["error"] = error
                log.warning(f"[{index}/{len(entries)}] failed (will retry next run): {url} | {error}")
            else:
                record["http_status"] = resp.status_code
                record["final_url"] = resp.url
                record["content_type"] = resp.headers.get("Content-Type", "")
                if resp.status_code == 200:
                    html_name = f"{key}.html.gz"
                    write_atomic(html_dir / html_name, gzip.compress(resp.content))  # raw bytes, untouched
                    record["outcome"] = "saved"
                    record["bytes"] = len(resp.content)
                    record["html_file"] = html_name
                    log.info(f"[{index}/{len(entries)}] saved: {url}")
                elif resp.status_code == 429 or resp.status_code >= 500:
                    record["outcome"] = "failed"
                    record["error"] = error
                    log.warning(f"[{index}/{len(entries)}] failed (will retry next run): {url} | {error}")
                else:
                    record["outcome"] = "http_error"
                    log.warning(f"[{index}/{len(entries)}] HTTP {resp.status_code}: {url}")

            # The receipt is written last, after the page file, so a receipt always
            # means "the page file is there too".
            write_receipt(meta_path, record)

    finally:
        counts = rebuild_manifest(meta_dir, out / "manifest.csv")
        log.info(f"Run summary: attempted {attempted}, skipped as already done {skipped_done}")
        log.info("Manifest outcomes so far: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if limit_hit:
            log.info("Stopped at the --limit. Run again (or use --all) to continue.")


if __name__ == "__main__":
    main()