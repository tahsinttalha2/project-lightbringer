"""
Thin adapter between the ingestion pipeline and your actual ASR /
classification models. Real logic belongs in models.py — this file
just calls it if it's implemented, and falls back to an honest stub
label if it isn't, so the pipeline never silently fabricates a result.
"""
import models


def run_asr(audio_path: str) -> str:
    fn = getattr(models, "transcribe", None)
    if callable(fn):
        try:
            return fn(audio_path)
        except Exception as e:
            return f"[ASR error] {e}"
    return "[ASR stub — implement transcribe(audio_path) in models.py]"


def classify_text(text: str) -> dict:
    fn = getattr(models, "classify", None)
    if callable(fn):
        try:
            return fn(text)
        except Exception as e:
            return {"label": "error", "confidence": None, "note": str(e)}
    return {
        "label": "stub",
        "confidence": None,
        "note": "Implement classify(text) in models.py to get a real verdict",
    }