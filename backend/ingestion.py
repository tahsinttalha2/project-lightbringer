#!/usr/bin/env python3
"""
Bengali YouTube ingestion engine (single-file version).

Reads a .txt file of YouTube URLs (one per line), and for each video:
  1. Fetches metadata (title, description, channel, upload date, subscriber
     count, views, likes) via yt-dlp.
  2. Fetches a transcript, but ONLY a manually-uploaded one -- never an
     auto-generated/ASR caption track (enforced structurally: only the
     `subtitles` dict is ever read, never `automatic_captions`).
  3. If no manual transcript exists, downloads the audio and runs it
     through the tituFC NeMo ASR model to produce one.
  4. Validates the video is actually Bengali-language content, based on
     the Bengali-Unicode-script ratio of the transcript text -- not on
     metadata alone, since titles/descriptions can be misleading.
  5. Writes one row per video to a CSV, resumable across runs.
  6. In a second pass, runs every transcript that (a) came from ASR, or
     (b) came from a manual caption track with no punctuation at all,
     through a small Gemma 4 model to restore punctuation --
     deterministically (greedy decoding) and with a word-preservation
     check that rejects any output where the model changed wording.

Every run also writes a JSON manifest recording exact configuration, model
ids, and installed library versions, for reproducibility.

Usage:
    python bengali_youtube_ingestion.py --urls urls.txt --output-dir ./data
    python bengali_youtube_ingestion.py --urls urls.txt --output-dir ./data --skip-cleaning
    python bengali_youtube_ingestion.py --urls urls.txt --output-dir ./data --force

This file was originally split into ingestion.py / asr_engine.py /
text_utils.py / clean_transcript.py / pipeline.py for readability; it is
merged here into one script, in the same top-to-bottom order, at the
person's request. Section banners below mark where each former module
starts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from importlib import metadata as _importlib_metadata

import pandas as pd
import soundfile
import torch
import yt_dlp

# ======================================================================= #
# SECTION 1 -- text utilities (formerly text_utils.py)
#
# Deterministic, dependency-light text utilities for the Bengali YouTube
# ingestion pipeline.
#
# Design principle (reproducibility): every check here is a fixed,
# documented rule -- Unicode code-point ranges and character-class
# membership -- not a statistical/probabilistic language-ID model. That
# keeps the "is this Bengali?", "does this have punctuation?" and "did the
# LLM change any words?" decisions fully deterministic and auditable
# across re-runs, which matters for a dataset meant to accompany a journal
# submission.
# ======================================================================= #

# Bengali (Bangla) Unicode block: U+0980-U+09FF
_BENGALI_LOW, _BENGALI_HIGH = 0x0980, 0x09FF

# Bengali sentence-final marks (danda / double danda) + the Latin punctuation
# that also shows up routinely in Bengali web/caption text.
_PUNCT_CHARS = set("।॥,.!?;:—–-\"'()[]{}…“”‘’")

# Same set, used as a regex character class to strip punctuation for the
# word-preservation check.
_PUNCT_STRIP_RE = re.compile(r"[।॥,.!?;:—–\-\"'()\[\]{}…“”‘’]")
_WORD_SPLIT_RE = re.compile(r"\s+")


def bengali_script_ratio(text: str) -> float:
    """Fraction of *alphabetic* characters in `text` that fall inside the
    Bengali Unicode block. Digits, whitespace and punctuation are ignored
    (they are script-neutral and would otherwise dilute the signal).
    Returns 0.0 for empty text or text with no alphabetic characters at all.
    """
    if not text:
        return 0.0
    total = 0
    bengali = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if _BENGALI_LOW <= ord(ch) <= _BENGALI_HIGH:
                bengali += 1
    return (bengali / total) if total else 0.0


def is_bengali_text(text: str, threshold: float = 0.6, min_letters: int = 10) -> bool:
    """A text is treated as Bengali iff it has at least `min_letters`
    alphabetic characters AND at least `threshold` of them fall in the
    Bengali Unicode block. Short/empty strings default to False (we do not
    guess on too little evidence) rather than passing by default.
    """
    if not text:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < min_letters:
        return False
    return bengali_script_ratio(text) >= threshold


def has_punctuation(text: str) -> bool:
    """True iff the text contains at least one sentence-level punctuation
    mark (Bengali danda/double-danda or standard Latin punctuation). This is
    a presence check only -- it flags raw, unpunctuated ASR-style output vs.
    normally punctuated text; it does not grade punctuation quality.
    """
    if not text:
        return False
    return any(ch in _PUNCT_CHARS for ch in text)


def strip_punctuation_and_normalize(text: str) -> list[str]:
    """Remove punctuation, NFC-normalize, collapse whitespace, and return
    the resulting word list. Used only to *compare* word sequences before
    and after LLM cleaning -- never to alter what actually gets saved to
    the CSV.
    """
    if not text:
        return []
    stripped = _PUNCT_STRIP_RE.sub(" ", text)
    stripped = unicodedata.normalize("NFC", stripped)
    return [w for w in _WORD_SPLIT_RE.split(stripped.strip()) if w]


def words_preserved(original: str, candidate: str) -> tuple[bool, str]:
    """Anti-hallucination guardrail for the LLM cleaning stage.

    Validates that `candidate` (the LLM's output) contains *exactly* the
    same word sequence as `original`, modulo punctuation/whitespace/casing
    differences introduced by normalization. If the LLM added, dropped,
    reordered, translated or substituted even a single word, this returns
    False and the caller must discard the LLM output rather than trust it.

    Returns (ok, reason_string).
    """
    orig_words = strip_punctuation_and_normalize(original)
    cand_words = strip_punctuation_and_normalize(candidate)
    if len(orig_words) != len(cand_words):
        return False, f"word count changed: {len(orig_words)} -> {len(cand_words)}"
    for i, (a, b) in enumerate(zip(orig_words, cand_words)):
        if a != b:
            return False, f"word mismatch at position {i}: '{a}' -> '{b}'"
    return True, "ok"


def chunk_text_by_words(text: str, max_words: int = 120) -> list[str]:
    """Split a transcript into word-count-bounded chunks.

    We chunk by raw word count rather than by sentence, because raw
    ASR / unpunctuated manual-caption text has no sentence boundaries yet
    (that is exactly what the cleaning stage is meant to add). Keeping
    chunks short does two things: it keeps each call within a small
    context window (important on a 6 GB card), and it gives the LLM less
    room to drift or hallucinate per call, which the word-preservation
    check then verifies.
    """
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


# ======================================================================= #
# SECTION 2 -- YouTube data collection (formerly ingestion.py)
#
# Built on yt-dlp rather than the official YouTube Data API. Reasoning:
# - yt-dlp needs no API key/quota and returns view_count, like_count,
#   channel_follower_count, upload_date, and both `subtitles` (manually
#   uploaded) and `automatic_captions` (ASR-generated) in one place.
# - YouTube does not expose dislike counts publicly through any API --
#   that field is intentionally absent here, not an oversight.
# - Manual vs. auto-generated caption tracks are genuinely distinguishable:
#   yt-dlp separates them into two different dict keys itself, so filtering
#   out ASR captions is just "only look in `subtitles`, never in
#   `automatic_captions`".
# ======================================================================= #

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
    views, likes, upload date, and the *list* of available caption tracks
    (not their content yet)."""
    with yt_dlp.YoutubeDL(_base_opts()) as ydl:
        info = ydl.extract_info(url, download=False)
    return info


def fetch_comments(url: str, max_comments: int = 200) -> list[dict]:
    """Separate, heavier call -- comment pagination is what actually makes
    this slow on popular videos. Not used by the CSV pipeline by default,
    kept here for anyone extending the dataset later."""
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
    back to `automatic_captions` -- that's the filtering requirement,
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
                "Only an auto-generated (ASR) Bangla caption track exists -- "
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


def _fmt_upload_date(raw: str | None) -> str | None:
    """yt-dlp gives upload_date as 'YYYYMMDD'; normalize to ISO for the CSV."""
    if not raw or len(raw) != 8:
        return raw
    try:
        return _dt.datetime.strptime(raw, "%Y%m%d").date().isoformat()
    except ValueError:
        return raw


def extract_metadata_fields(info: dict) -> dict:
    """Normalize a raw yt-dlp info dict into the flat set of fields the
    pipeline writes to the CSV. Kept as a separate function so the CSV
    schema and the yt-dlp field mapping stay in one auditable place."""
    return {
        "video_id": info.get("id"),
        "url": info.get("webpage_url") or info.get("original_url"),
        "title": info.get("title"),
        "description": info.get("description"),
        "channel_name": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "upload_date": _fmt_upload_date(info.get("upload_date")),
        "duration_seconds": info.get("duration"),
        "subscriber_count": info.get("channel_follower_count"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        # Raw yt-dlp-reported language, when the uploader/YouTube set one.
        # Frequently absent -- treated only as a weak, auxiliary signal.
        # The authoritative language check is the transcript-level
        # Bengali-script-ratio test in is_bengali_text() (Section 1).
        "metadata_language_field": info.get("language"),
    }


# ======================================================================= #
# SECTION 3 -- ASR engine (formerly asr_engine.py)
#
# Reusable NeMo ASR engine, refactored from the original tituFC_asr.py so
# the model is loaded once and reused for every video in a batch, rather
# than loaded and deleted on every single call. The decoding configuration
# (local relative-positional attention, `greedy_batch` strategy) is kept
# identical to the original script so ASR output stays consistent with any
# prior single-file runs. `greedy_batch` is already deterministic (no
# sampling), which the reproducibility goal of this pipeline requires.
# ======================================================================= #

DEFAULT_ASR_MODEL_ID = "hishab/titu_stt_bn_fastconformer"

asr_logger = logging.getLogger("asr_engine")


def ensure_mono_wav(audio_path: str) -> str:
    """NeMo/soundfile expect mono audio. If the source is stereo, downmix
    and write a `_mono.wav` sibling file; otherwise return the path
    unchanged. Downmixing by averaging channels is deterministic."""
    data, sr = soundfile.read(audio_path)
    if data.ndim == 1:
        return audio_path
    mono_path = audio_path.replace(".wav", "_mono.wav")
    soundfile.write(mono_path, data.mean(axis=1), sr)
    return mono_path


class ASREngine:
    """Thin, stateful wrapper around a NeMo ASRModel.

    Usage:
        with ASREngine(model_id=..., device=...) as asr:
            text = asr.transcribe("/path/to/audio.wav")
    or manually:
        asr = ASREngine(...)
        asr.load()
        ...
        asr.unload()
    """

    def __init__(self, model_id: str = DEFAULT_ASR_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self.model_revision: str | None = None  # filled in on load(), for the manifest

    def load(self) -> None:
        if self._model is not None:
            return
        import nemo.collections.asr as NEMO  # heavy + optional; import lazily

        asr_logger.info("Loading ASR model '%s' on %s", self.model_id, self.device)
        model = NEMO.models.ASRModel.from_pretrained(model_name=self.model_id)
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn", att_context_size=[128, 128]
        )
        model.change_subsampling_conv_chunking_factor(1)

        model.cfg.decoding.strategy = "greedy_batch"  # deterministic, no sampling
        model.change_decoding_strategy(model.cfg.decoding)

        self._model = model.to(self.device)
        self._model.eval()

    def transcribe(self, audio_path: str) -> str:
        if self._model is None:
            self.load()
        audio_path = ensure_mono_wav(audio_path)
        with torch.inference_mode():
            output = self._model.transcribe([audio_path])
        item = output[0][0] if isinstance(output, tuple) else output[0]
        text = item.text if hasattr(item, "text") else str(item)
        return text.strip()

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        asr_logger.info("ASR model unloaded, VRAM released.")

    def __enter__(self) -> "ASREngine":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()


# ======================================================================= #
# SECTION 4 -- Gemma-based punctuation cleaning (formerly clean_transcript.py)
#
# Punctuation/cleanup stage for transcripts that either (a) came out of the
# ASR model with no punctuation at all, or (b) are manually-uploaded
# YouTube captions that happen to lack punctuation. Uses a small Gemma 4
# edge model (google/gemma-4-E2B-it by default) purely as a *punctuation
# restorer*, not a free-form rewriter.
#
# Reproducibility / anti-hallucination design:
#   1. Greedy decoding only (do_sample=False, num_beams=1) -- no
#      temperature, no top-k/top-p sampling, so generation is deterministic
#      given the same weights, input and library versions.
#   2. The transcript is chunked into short word-bounded pieces
#      (chunk_text_by_words, Section 1) so the model never has to hold a
#      whole long transcript's structure in its head at once.
#   3. Every chunk's output is checked with words_preserved() (Section 1)
#      against the *original* chunk. If the model added, dropped, reordered
#      or substituted a single word, the output is rejected outright. A
#      rejected chunk is retried once with a stricter reminder prompt; if
#      it fails again, cleaning for that whole transcript is marked failed
#      and the raw transcript is kept as-is.
#   4. Model id + revision + generation config + library versions are all
#      logged into the run manifest by the pipeline section below.
#
# VRAM note: on an official 6 GB card, even the smallest Gemma 4 edge model
# (E2B) in 4-bit is tight -- community NF4 benchmarks report ~6.3 GB just
# for weights+activations at modest sequence lengths. This module therefore
# defaults to NF4 + double quantization, keeps chunks short to bound the KV
# cache, and will retry a failing chunk on CPU rather than crash the whole
# run if a single chunk hits CUDA OOM.
# ======================================================================= #

DEFAULT_CLEAN_MODEL_ID = "google/gemma-4-E2B-it"

clean_logger = logging.getLogger("clean_transcript")

_SYSTEM_INSTRUCTION = (
    "You are a strict punctuation-restoration tool for Bengali (Bangla) "
    "speech-to-text transcripts. You will be given a raw transcript with "
    "little or no punctuation and inconsistent spacing.\n\n"
    "Rules, followed exactly:\n"
    "1. Add appropriate punctuation (Bengali দাঁড়ি '।', commas, question "
    "marks, etc.) and fix spacing/casing only.\n"
    "2. Do NOT add, remove, reorder, translate, summarize or reword ANY "
    "word. The word sequence must stay 100% identical to the input.\n"
    "3. Do NOT add any commentary, explanation, labels, or markdown.\n"
    "4. Output ONLY the corrected transcript text, nothing else."
)

_RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous attempt changed the wording. Return the "
    "exact same words, in the exact same order -- only insert punctuation "
    "and fix spacing/casing."
)


@dataclass
class ChunkResult:
    original: str
    cleaned: str | None
    ok: bool
    reason: str
    device_used: str = "unknown"


@dataclass
class CleaningResult:
    status: str  # "success" | "failed_validation" | "error" | "not_required"
    cleaned_text: str | None
    chunks: list[ChunkResult] = field(default_factory=list)
    error: str | None = None


class GemmaPunctuationRestorer:
    def __init__(
        self,
        model_id: str = DEFAULT_CLEAN_MODEL_ID,
        device: str | None = None,
        chunk_words: int = 120,
        seed: int = 42,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_words = chunk_words
        self.seed = seed
        self._model = None
        self._tokenizer = None
        self.model_revision: str | None = None  # set on load(), for the manifest

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self._model is not None:
            return

        torch.manual_seed(self.seed)  # no effect on greedy decoding itself,
        # kept for completeness / in case anyone later ablates with sampling

        from transformers import AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        clean_logger.info("Loading cleaning model '%s' (4-bit NF4)", self.model_id)

        # Gemma 4 is a very recently released, natively multimodal family;
        # the exact AutoModel class exposed by `transformers` for text-only
        # use has varied across community write-ups. Try the plausible
        # options in order rather than hard-failing on one guess. If none
        # of these work against your installed `transformers` version,
        # check the model card at
        # https://huggingface.co/google/gemma-4-E2B-it for the current
        # loading snippet and update this method accordingly.
        model = None
        last_err: Exception | None = None
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                import transformers as _tf
                cls = getattr(_tf, cls_name)
                model = cls.from_pretrained(
                    self.model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                )
                clean_logger.info("Loaded cleaning model via transformers.%s", cls_name)
                break
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, we try alternatives
                last_err = exc
                continue
        if model is None:
            raise RuntimeError(
                f"Could not load '{self.model_id}' with any known AutoModel "
                f"class. Last error: {last_err}"
            )

        self._model = model
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        try:
            self.model_revision = getattr(self._model.config, "_commit_hash", None)
        except Exception:  # noqa: BLE001
            self.model_revision = None

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clean_logger.info("Cleaning model unloaded, VRAM released.")

    def __enter__(self) -> "GemmaPunctuationRestorer":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate(self, text: str, retry: bool = False) -> str:
        instruction = _SYSTEM_INSTRUCTION + (_RETRY_SUFFIX if retry else "")
        messages = [
            {"role": "user", "content": instruction + "\n\nTranscript:\n" + text}
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]
        # Generous but bounded headroom: punctuation adds few tokens, so the
        # cleaned chunk should be close in length to the input. Bounding
        # max_new_tokens also caps worst-case KV-cache growth on a tight
        # VRAM budget, and a truncated generation will simply fail the
        # word-preservation check downstream rather than corrupt the data.
        max_new_tokens = int(input_len * 1.4) + 32

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy -- required for determinism
                num_beams=1,       # pure greedy, not beam search
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = output_ids[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _clean_chunk(self, chunk: str) -> ChunkResult:
        for attempt, retry in enumerate((False, True)):
            try:
                candidate = self._generate(chunk, retry=retry)
            except torch.cuda.OutOfMemoryError:
                clean_logger.warning(
                    "CUDA OOM cleaning a %d-word chunk; retrying that chunk on CPU.",
                    len(chunk.split()),
                )
                torch.cuda.empty_cache()
                original_device = self._model.device
                try:
                    self._model.to("cpu")
                    candidate = self._generate(chunk, retry=retry)
                finally:
                    self._model.to(original_device)
                ok, reason = words_preserved(chunk, candidate)
                return ChunkResult(chunk, candidate if ok else None, ok, reason, "cpu_fallback")
            except Exception as exc:  # noqa: BLE001
                return ChunkResult(chunk, None, False, f"generation error: {exc}")

            ok, reason = words_preserved(chunk, candidate)
            if ok:
                return ChunkResult(chunk, candidate, True, reason, str(self._model.device))
            clean_logger.warning(
                "Chunk failed word-preservation check (attempt %d): %s",
                attempt + 1, reason,
            )
        # both attempts failed
        return ChunkResult(chunk, None, False, reason)

    def clean(self, transcript: str) -> CleaningResult:
        if self._model is None:
            self.load()
        if not transcript or not transcript.strip():
            return CleaningResult(status="error", cleaned_text=None, error="empty transcript")

        chunks = chunk_text_by_words(transcript, max_words=self.chunk_words)
        results: list[ChunkResult] = []
        for chunk in chunks:
            results.append(self._clean_chunk(chunk))

        if all(r.ok for r in results):
            cleaned = " ".join(r.cleaned for r in results)
            return CleaningResult(status="success", cleaned_text=cleaned, chunks=results)

        failed = [r for r in results if not r.ok]
        return CleaningResult(
            status="failed_validation",
            cleaned_text=None,
            chunks=results,
            error=f"{len(failed)}/{len(results)} chunk(s) failed word-preservation check",
        )


# ======================================================================= #
# SECTION 5 -- pipeline orchestrator (formerly pipeline.py)
# ======================================================================= #

CSV_COLUMNS = [
    "video_id", "url", "title", "channel_name", "channel_id", "upload_date",
    "duration_seconds", "subscriber_count", "view_count", "like_count",
    "metadata_language_field", "title_desc_bengali_ratio",
    "language_check_status",
    "transcript_source", "transcript", "transcript_bengali_ratio",
    "has_punctuation", "needs_cleaning",
    "cleaning_status", "cleaned_transcript", "cleaning_notes",
    "asr_model_id", "cleaning_model_id",
    "processed_at", "notes",
]

logger = logging.getLogger("pipeline")


# --------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------- #
def read_urls(path: str) -> list[str]:
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[CSV_COLUMNS]
    return pd.DataFrame(columns=CSV_COLUMNS)


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)  # atomic on POSIX


def upsert_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    row = {k: row.get(k) for k in CSV_COLUMNS}
    mask = df["video_id"] == row["video_id"]
    if mask.any():
        idx = df.index[mask][0]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def pkg_version(name: str) -> str | None:
    try:
        return _importlib_metadata.version(name)
    except _importlib_metadata.PackageNotFoundError:
        return None


# --------------------------------------------------------------------- #
# Manifest (reproducibility record)
# --------------------------------------------------------------------- #
def build_manifest(args: argparse.Namespace, urls_path: str, n_urls: int) -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    manifest = {
        "run_timestamp_utc": now_iso(),
        "input_urls_file": os.path.abspath(urls_path),
        "input_urls_sha256": sha256_file(urls_path),
        "input_url_count": n_urls,
        "config": {
            "asr_model_id": args.asr_model,
            "cleaning_model_id": args.clean_model,
            "bengali_threshold": args.bengali_threshold,
            "chunk_words": args.chunk_words,
            "seed": args.seed,
            "skip_cleaning": args.skip_cleaning,
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "library_versions": {
            name: pkg_version(name)
            for name in (
                "torch", "transformers", "accelerate", "bitsandbytes",
                "nemo_toolkit", "yt-dlp", "soundfile", "pandas",
            )
        },
        "decoding": {
            "asr_strategy": "greedy_batch (deterministic, no sampling)",
            "cleaning_strategy": "greedy (do_sample=False, num_beams=1)",
        },
        "language_filter": {
            "method": "Bengali Unicode block ratio (U+0980-U+09FF) over "
                      "alphabetic characters; rule-based, not a statistical "
                      "language-ID model",
            "threshold": args.bengali_threshold,
        },
    }
    try:
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
        manifest["pip_freeze"] = pip_freeze.splitlines()
    except Exception as exc:  # noqa: BLE001
        manifest["pip_freeze_error"] = str(exc)
    return manifest


# --------------------------------------------------------------------- #
# Phase A: metadata + transcript / ASR
# --------------------------------------------------------------------- #
def process_video(url: str, asr: ASREngine, audio_dir: str, cfg: argparse.Namespace) -> dict:
    row = {c: None for c in CSV_COLUMNS}
    row["url"] = url
    row["processed_at"] = now_iso()
    row["asr_model_id"] = cfg.asr_model
    row["cleaning_model_id"] = cfg.clean_model

    info = fetch_basic_info(url)
    row.update(extract_metadata_fields(info))

    title_desc = f"{info.get('title') or ''} {info.get('description') or ''}"
    td_ratio = bengali_script_ratio(title_desc)
    row["title_desc_bengali_ratio"] = round(td_ratio, 4)

    lang_field = (info.get("language") or "").lower()
    obviously_not_bengali = td_ratio < 0.3 and not lang_field.startswith("bn")
    if obviously_not_bengali:
        row["language_check_status"] = "excluded_non_bengali_pre_check"
        row["notes"] = (
            f"Skipped before download: title/description Bengali-script "
            f"ratio {td_ratio:.2f} < 0.30 and metadata language "
            f"'{info.get('language')}' is not Bengali."
        )
        row["transcript_source"] = "none"
        return row

    transcript_result = fetch_transcript(url, info=info)
    if transcript_result["source"] == "manual":
        row["transcript_source"] = "manual"
        row["transcript"] = transcript_result["text"]
        row["notes"] = transcript_result["note"]
    else:
        # No usable manual transcript -> fall back to ASR.
        row["notes"] = transcript_result["note"]
        try:
            audio_path = download_audio(url, audio_dir)
            transcript_text = asr.transcribe(audio_path)
            row["transcript_source"] = "asr"
            row["transcript"] = transcript_text
        except Exception as exc:  # noqa: BLE001
            row["transcript_source"] = "none"
            row["language_check_status"] = "error"
            row["notes"] = f"ASR/download failed: {exc}"
            return row

    transcript = row.get("transcript") or ""
    t_ratio = bengali_script_ratio(transcript)
    row["transcript_bengali_ratio"] = round(t_ratio, 4)

    if not is_bengali_text(transcript, threshold=cfg.bengali_threshold):
        row["language_check_status"] = "excluded_non_bengali"
        row["notes"] = (row["notes"] or "") + (
            f" | Transcript Bengali-script ratio {t_ratio:.2f} is below "
            f"threshold {cfg.bengali_threshold}; excluded."
        )
        row["needs_cleaning"] = False
        row["cleaning_status"] = "not_required"
        return row

    row["language_check_status"] = "pass"
    punct = has_punctuation(transcript)
    row["has_punctuation"] = punct
    needs_cleaning = (row["transcript_source"] == "asr") or (not punct)
    row["needs_cleaning"] = needs_cleaning
    row["cleaning_status"] = "pending" if needs_cleaning else "not_required"
    return row


def run_phase_a(urls: list[str], df: pd.DataFrame, csv_path: str, args: argparse.Namespace) -> pd.DataFrame:
    done_ids = set(df.loc[df["transcript_source"].notna(), "video_id"].dropna()) if not args.force else set()

    to_process = []
    for url in urls:
        try:
            vid = fetch_basic_info(url).get("id")
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not resolve video id for %s: %s", url, exc)
            continue
        if vid in done_ids:
            logger.info("Skipping already-processed video %s", vid)
            continue
        to_process.append(url)

    if not to_process:
        logger.info("Phase A: nothing new to process.")
        return df

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    with ASREngine(model_id=args.asr_model, device=args.device) as asr:
        for i, url in enumerate(to_process, 1):
            logger.info("[%d/%d] Processing %s", i, len(to_process), url)
            try:
                row = process_video(url, asr, audio_dir, args)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled error processing %s", url)
                row = {c: None for c in CSV_COLUMNS}
                row.update({
                    "url": url, "processed_at": now_iso(),
                    "language_check_status": "error", "transcript_source": "none",
                    "notes": f"Unhandled error: {exc}",
                })
            df = upsert_row(df, row)
            atomic_write_csv(df, csv_path)  # checkpoint after every video

    return df


# --------------------------------------------------------------------- #
# Phase B: Gemma-based punctuation cleaning
# --------------------------------------------------------------------- #
def run_phase_b(df: pd.DataFrame, csv_path: str, args: argparse.Namespace) -> pd.DataFrame:
    if args.skip_cleaning:
        logger.info("Phase B skipped (--skip-cleaning).")
        return df

    pending_mask = df["needs_cleaning"].astype(str).str.lower() == "true"
    if not args.force:
        pending_mask &= df["cleaning_status"].isin(["pending", "failed_validation", "error", None]) \
            | df["cleaning_status"].isna()
    rows_to_clean = df[pending_mask]

    if rows_to_clean.empty:
        logger.info("Phase B: nothing to clean.")
        return df

    logger.info("Phase B: cleaning %d transcript(s) with %s", len(rows_to_clean), args.clean_model)

    with GemmaPunctuationRestorer(
        model_id=args.clean_model, device=args.device,
        chunk_words=args.chunk_words, seed=args.seed,
    ) as restorer:
        for idx, r in rows_to_clean.iterrows():
            logger.info("Cleaning video_id=%s", r["video_id"])
            result = restorer.clean(r["transcript"] or "")
            df.at[idx, "cleaning_status"] = result.status
            df.at[idx, "cleaned_transcript"] = result.cleaned_text
            if result.status != "success":
                df.at[idx, "cleaning_notes"] = result.error
            atomic_write_csv(df, csv_path)  # checkpoint after every transcript

    return df


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bengali YouTube ingestion engine")
    p.add_argument("--urls", required=True, help="Path to .txt file, one YouTube URL per line")
    p.add_argument("--output-dir", required=True, help="Base directory for CSV, audio/, logs, manifest")
    p.add_argument("--csv-name", default="bengali_youtube_dataset.csv")
    p.add_argument("--asr-model", default=DEFAULT_ASR_MODEL_ID)
    p.add_argument("--clean-model", default=DEFAULT_CLEAN_MODEL_ID)
    p.add_argument("--bengali-threshold", type=float, default=0.6,
                    help="Minimum fraction of alphabetic characters that must be "
                         "Bengali-script for a transcript to be accepted (default 0.6)")
    p.add_argument("--chunk-words", type=int, default=120,
                    help="Max words per chunk sent to the cleaning model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    p.add_argument("--skip-cleaning", action="store_true", help="Only run Phase A (metadata/transcript/ASR)")
    p.add_argument("--force", action="store_true", help="Reprocess videos/rows already in the CSV")
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "pipeline.log")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )

    # Best-effort determinism. torch.use_deterministic_algorithms(True) is
    # deliberately NOT forced globally here: some ops used by NeMo/Gemma 4
    # don't have deterministic CUDA kernels and would hard-error rather than
    # just being slower. cudnn.deterministic still helps on the ops that do.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    urls = read_urls(args.urls)
    logger.info("Loaded %d URL(s) from %s", len(urls), args.urls)

    csv_path = os.path.join(args.output_dir, args.csv_name)
    df = load_existing_csv(csv_path)

    manifest = build_manifest(args, args.urls, len(urls))
    manifest_path = os.path.join(
        args.output_dir, f"manifest_{_dt.datetime.now().strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote reproducibility manifest to %s", manifest_path)

    df = run_phase_a(urls, df, csv_path, args)
    df = run_phase_b(df, csv_path, args)

    atomic_write_csv(df, csv_path)
    logger.info("Done. Dataset written to %s (%d rows).", csv_path, len(df))


if __name__ == "__main__":
    main()