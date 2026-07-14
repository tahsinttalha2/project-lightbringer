import subprocess
import sys
import os
import torch
import numpy as np
import soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor

script_dir = os.path.dirname(os.path.abspath(__file__))
asr_root = os.path.dirname(script_dir)
audio_dir = os.path.join(asr_root, "audio")
whisper_dir = os.path.join(asr_root, "whisper")

os.makedirs(audio_dir, exist_ok=True)
os.makedirs(whisper_dir, exist_ok=True)

_vad_model = None
_vad_utils = None


def _get_vad():
    """Lazy-load Silero VAD. Loading this at import time (as in the previous
    version) means the script can't even be imported without a network call
    and a torch.hub cache write -- and it crashes on import if offline."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        print("[*] loading silero-vad (first run will download it) ...")
        _vad_model, _vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False
        )
    return _vad_model, _vad_utils


def _run_subprocess(command_arguments, tool_name):
    try:
        subprocess.run(command_arguments, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"'{tool_name}' is not installed or not on PATH. "
            f"Install it before running this script."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{tool_name} failed with exit code {e.returncode}")


def download_audio(youtube_video_url: str, output_file_path: str = None) -> str:
    if output_file_path is None:
        output_file_path = os.path.join(audio_dir, "audio.wav")

    print(f"[*] downloading audio from {youtube_video_url} ...")
    command_arguments = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", output_file_path.replace(".wav", ".%(ext)s"),
        youtube_video_url,
    ]
    _run_subprocess(command_arguments, "yt-dlp")

    if not os.path.exists(output_file_path):
        base_filename = os.path.basename(output_file_path.replace(".wav", ""))
        search_dir = os.path.dirname(output_file_path) or "."
        matching_files = [
            os.path.join(search_dir, f)
            for f in os.listdir(search_dir)
            if f.startswith(base_filename)
        ]
        if matching_files:
            # prefer the most recently written match, not just the first
            # one os.listdir happens to return (order is not guaranteed)
            matching_files.sort(key=os.path.getmtime, reverse=True)
            output_file_path = matching_files[0]
        else:
            raise FileNotFoundError("could not locate downloaded audio file.")

    print(f"[ok] audio downloaded: {output_file_path}")
    return output_file_path


def trim_audio(input_file_path: str, duration_in_seconds: float, output_file_path: str = None) -> str:
    if output_file_path is None:
        output_file_path = os.path.join(audio_dir, "trimmed.wav")

    print(f"[*] trimming audio to first {duration_in_seconds} seconds ...")
    command_arguments = [
        "ffmpeg", "-y",
        "-i", input_file_path,
        "-t", str(duration_in_seconds),
        "-ar", "16000",
        "-ac", "1",
        output_file_path,
    ]
    _run_subprocess(command_arguments, "ffmpeg")
    print(f"[ok] trimmed audio saved: {output_file_path}")
    return output_file_path


def _read_wav_as_tensor(audio_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """Read a wav file with soundfile and return a mono float32 torch tensor.
    Deliberately avoids torchaudio for I/O -- torchaudio>=2.9 dropped its
    built-in audio backends in favor of the separate torchcodec package,
    and the silero-vad repo's own read_audio() helper hits the same wall
    since it also shells out to torchaudio. soundfile has no such
    dependency and reads wav natively."""
    audio_array, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # collapse to mono if needed
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"expected {target_sample_rate}Hz audio, got {sample_rate}Hz. "
            f"trim_audio() should already resample to 16kHz -- check the ffmpeg step."
        )
    return torch.from_numpy(audio_array)


def strip_silence(audio_path: str, output_path: str) -> str:
    """VAD-trim silence/dead air before chunked transcription. This is the
    real fix for chunk-boundary hallucination, not just the generate_kwargs
    guards -- silence is what triggers Whisper to invent text in the first
    place."""
    print(f"[*] stripping silence from {audio_path} ...")
    vad_model, vad_utils = _get_vad()
    get_speech_timestamps, *_ = vad_utils

    wav = _read_wav_as_tensor(audio_path, target_sample_rate=16000)
    timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    if not timestamps:
        raise ValueError(f"no speech detected in {audio_path}")

    speech = torch.cat([wav[t["start"]:t["end"]] for t in timestamps])
    sf.write(output_path, speech.numpy(), 16000, subtype="PCM_16")
    print(f"[ok] silence-stripped audio saved: {output_path}")
    return output_path


def transcribe_audio(audio_file_path: str, hf_model_id: str = "bangla-speech-processing/BanglaASR"):
    hardware_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] hardware device: {hardware_device}")

    if hardware_device == "cpu":
        print("[warning] running on cpu -- this will be computationally expensive.")

    # fp16 on unsupported/older GPUs can silently produce NaNs instead of
    # erroring. Only use fp16 if the device actually advertises support.
    use_fp16 = hardware_device != "cpu" and torch.cuda.is_bf16_supported() is not None
    compute_dtype = torch.float16 if use_fp16 else torch.float32

    print(f"[*] loading huggingface whisper model: {hf_model_id} ...")
    try:
        try:
            safe_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            safe_feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_id)
            print("[*] using model's native tokenizer/feature extractor")
        except Exception as native_error:
            print(f"[!] native tokenizer/feature extractor failed to load: {native_error}")
            print("[*] falling back to openai/whisper-medium base components")
            clean_base_reference = "openai/whisper-medium"
            safe_tokenizer = AutoTokenizer.from_pretrained(clean_base_reference)
            safe_feature_extractor = AutoFeatureExtractor.from_pretrained(clean_base_reference)

        transcriber = pipeline(
            "automatic-speech-recognition",
            model=hf_model_id,
            tokenizer=safe_tokenizer,
            feature_extractor=safe_feature_extractor,
            device=hardware_device,
            torch_dtype=compute_dtype,
        )
        transcriber.model.generation_config.forced_decoder_ids = None
    except Exception as loading_error:
        print(f"[!] failed to load huggingface model: {loading_error}")
        raise

    print(f"[*] transcribing ...")
    try:
        transcription_result = transcriber(
            audio_file_path,
            chunk_length_s=30,
            stride_length_s=5,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={
                "language": "bengali",
                "task": "transcribe",
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "condition_on_prev_tokens": False,  
                "no_repeat_ngram_size": 3,              
                "compression_ratio_threshold": 2.4,     
                "logprob_threshold": -1.0,              
                "no_speech_threshold": 0.6,             
            },
        )
    except torch.cuda.OutOfMemoryError as memory_error:
        print(f"[!] cuda out of memory: {memory_error}")
        raise
    except Exception as execution_error:
        print(f"[!] transcription failed: {execution_error}")
        raise

    if isinstance(transcription_result, list):
        transcription_result = transcription_result[0]

    formatted_result = {
        "text": transcription_result.get("text", ""),
        "segments": [],
    }

    for chunk in transcription_result.get("chunks", []):
        start_time = chunk.get("timestamp", (0.0, 0.0))[0]
        end_time = chunk.get("timestamp", (0.0, 0.0))[1]

        start_time = start_time if start_time is not None else 0.0
        end_time = end_time if end_time is not None else start_time + 1.0

        formatted_result["segments"].append({
            "start": start_time,
            "end": end_time,
            "text": chunk.get("text", ""),
        })

    # hallucination diagnostic runs AFTER the result is unwrapped/formatted,
    # not before -- the previous version ran it on the raw (possibly still
    # list-wrapped) result and printed nothing useful
    for i, segment in enumerate(formatted_result["segments"]):
        words = segment["text"].split()
        if len(words) > 3 and len(set(words)) / len(words) < 0.3:
            print(f"[hallucination suspect] segment {i} ({segment['start']:.2f}s): {segment['text'][:100]}")

    return formatted_result


def save_transcription_outputs(transcription_result, output_file_prefix: str):
    flat_text_path = os.path.join(whisper_dir, f"{output_file_prefix}.txt")
    segments_text_path = os.path.join(whisper_dir, f"{output_file_prefix}_segments.txt")

    with open(flat_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription_result["text"].strip())

    with open(segments_text_path, "w", encoding="utf-8") as segments_file:
        for segment in transcription_result["segments"]:
            segments_file.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'].strip()}\n")

    print(f"\nsaved flat asr transcript ->     {flat_text_path}")
    print(f"saved timestamped asr transcript -> {segments_text_path}")


def main():
    if len(sys.argv) < 3:
        print("usage: python asr_whisper.py <youtube_url> <duration_seconds> [output_prefix] [hf_model_id]")
        sys.exit(1)

    youtube_video_url = sys.argv[1]
    video_duration_seconds = float(sys.argv[2])
    output_file_prefix = sys.argv[3] if len(sys.argv) > 3 else "whisper_output"
    hf_model_id = sys.argv[4] if len(sys.argv) > 4 else "bangla-speech-processing/BanglaASR"

    full_audio_file = download_audio(youtube_video_url)

    trimmed_audio_path = os.path.join(audio_dir, f"{output_file_prefix}_trimmed.wav")
    trimmed_audio_file = trim_audio(
        full_audio_file, video_duration_seconds, output_file_path=trimmed_audio_path
    )

    # strip_silence existed in the previous version but was never called --
    # it's wired into the pipeline now, which is the actual chunking fix
    silence_stripped_path = os.path.join(audio_dir, f"{output_file_prefix}_vad.wav")
    final_audio_file = strip_silence(trimmed_audio_file, silence_stripped_path)

    result = transcribe_audio(final_audio_file, hf_model_id=hf_model_id)
    save_transcription_outputs(result, output_file_prefix)


if __name__ == "__main__":
    main()