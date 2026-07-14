"""
All YouTube data collection lives here, built on yt-dlp rather than the
official YouTube Data API. Reasoning:

- yt-dlp needs no API key/quota and returns view_count, like_count,
  channel_follower_count, comments, and both `subtitles` (manually
  uploaded) and `automatic_captions` (ASR-generated) in one place.
- YouTube does not expose dislike counts publicly through any API —
  that field is intentionally absent here, not an oversight.
- Manual vs. auto-generated caption tracks are genuinely distinguishable:
  yt-dlp separates them into two different dict keys itself, so
  filtering out ASR captions is just "only look in `subtitles`, never
  in `automatic_captions`".

Every function here is synchronous and blocking (that's what yt-dlp is).
The FastAPI layer runs these via asyncio.to_thread so they don't block
the event loop.
"""
import os
import yt_dlp

BANGLA_LANG_CODES = ("bn", "bn-BD", "bn-IN")


def _base_opts(extra: dict | None = None) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    if extra:
        opts.update(extra)
    return opts


def fetch_basic_info(url: str) -> dict:
    """Single fast call: title, description, channel, subscriber count,
    views, likes, and the *list* of available caption tracks (not their
    content yet)."""
    with yt_dlp.YoutubeDL(_base_opts()) as ydl:
        info = ydl.extract_info(url, download=False)
    return info


def fetch_comments(url: str, max_comments: int = 200) -> list[dict]:
    """Separate, heavier call — comment pagination is what actually
    makes this slow on popular videos."""
    opts = _base_opts({
        "getcomments": True,
        "extractor_args": {"youtube": {"max_comments": [str(max_comments)]}},
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("comments") or []


def download_audio(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")
    opts = _base_opts({
        "skip_download": False,
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    video_id = info["id"]
    return os.path.join(out_dir, f"{video_id}.wav")


def fetch_transcript(url: str, info: dict | None = None) -> dict:
    """Only ever reads from `subtitles` (manually uploaded). Never falls
    back to `automatic_captions` — that's the filtering requirement,
    enforced structurally rather than by a keyword check."""
    if info is None:
        info = fetch_basic_info(url)

    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    lang = next((code for code in BANGLA_LANG_CODES if code in manual), None)
    if lang is None:
        has_auto_bn = any(code in auto for code in BANGLA_LANG_CODES)
        return {
            "source": "none",
            "text": None,
            "note": (
                "Only an auto-generated (ASR) Bangla caption track exists — "
                "excluded per filtering rule."
                if has_auto_bn else
                "No Bangla caption track, manual or auto-generated, was found."
            ),
        }

    opts = _base_opts({
        "writesubtitles": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "skip_download": True,
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        # yt-dlp writes the subtitle file to disk; we point outtmpl at /tmp
        ydl.params["outtmpl"] = {"default": "/tmp/%(id)s.%(ext)s"}
        result_info = ydl.extract_info(url, download=True)

    vtt_path = f"/tmp/{result_info['id']}.{lang}.vtt"
    text = None
    if os.path.exists(vtt_path):
        text = _vtt_to_plain_text(vtt_path)
        os.remove(vtt_path)

    return {"source": "manual", "text": text, "note": f"Manual '{lang}' transcript found"}


def _vtt_to_plain_text(path: str) -> str:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("WEBVTT", "NOTE")) or "-->" in line or line.isdigit():
                continue
            lines.append(line)
    # collapse consecutive duplicate lines (vtt often repeats cues)
    dedup = []
    for line in lines:
        if not dedup or dedup[-1] != line:
            dedup.append(line)
    return " ".join(dedup)