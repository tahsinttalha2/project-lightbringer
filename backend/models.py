"""
Plug your real models in here. This file is intentionally left as a thin,
empty shell — models_pipeline.py will import it and call these two
functions if they exist. Nothing else in the backend needs to change.

Implement whichever of these you're ready to wire up. Leave the other
unimplemented (or raise NotImplementedError) and the pipeline will fall
back to a labeled stub value instead of crashing the request.

Example wiring (uncomment and adapt):

    import torch
    from transformers import pipeline as hf_pipeline

    _asr = None
    _classifier = None

    def _load_asr():
        global _asr
        if _asr is None:
            # e.g. your fine-tuned Titu FastConformer checkpoint, or Whisper
            _asr = hf_pipeline("automatic-speech-recognition", model="path/to/your/asr-checkpoint")
        return _asr

    def transcribe(audio_path: str) -> str:
        asr = _load_asr()
        result = asr(audio_path)
        return result["text"]

    def _load_classifier():
        global _classifier
        if _classifier is None:
            _classifier = hf_pipeline("text-classification", model="path/to/your/banglabert-checkpoint")
        return _classifier

    def classify(text: str) -> dict:
        clf = _load_classifier()
        out = clf(text, truncation=True)[0]
        return {"label": out["label"], "confidence": round(out["score"], 4)}
"""

# def transcribe(audio_path: str) -> str:
#     ...

# def classify(text: str) -> dict:
#     ...