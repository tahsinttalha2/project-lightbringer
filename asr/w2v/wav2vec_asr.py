import os

import torch
from pyannote.audio import Pipeline as PIPELINE
import gc

import soundfile
from transformers import pipeline as TRANS_PIPELINE

import sys

ASR = "tituFC"

# dynamic directory paths for consistent file operation
script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(script_dir)
audio_dir = os.path.join(root, "audio")
output_dir = os.path.join(root, "output")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# implementing speaker diarisation
def speaker_diarisation(audio_src: str, device: str) -> list:
    # instantiate the diarization pipeline
    diarisation_pipeline = PIPELINE.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token = True
    )
    diarisation_pipeline.to(device)

    diarisation_output = diarisation_pipeline(audio_src)

    # map the accoustic landscape of the audio
    annotation = diarisation_output.speaker_diarization

    speaker = []

    # extract the start time, end time, and speaker label
    for speech_turn, _, speaker_label in annotation.itertracks(yield_label = True):
        speaker.append({
            "start_time": speech_turn.start,
            "end_time": speech_turn.end,
            "speaker": speaker_label
        })

    print("Speaker diarisation complete.\n")
    print("Attempting to perform memory cleanup...")

    # hardware cleanup to save vram
    del diarisation_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("Memory successfully cleared!\n")

    return speaker

# slicing audio into managable waves
def chunking(audio_src: str, speakers: list) -> list:
    audio_data, sampling_rate = soundfile.read(audio_src)
    total_audio_samples = len(audio_data)

    chunk_metadata_list = []

    for segment_index, segment_dictionary in enumerate(speakers):
        start_sample_index = int(segment_dictionary["start_time"] * sampling_rate)
        end_sample_index = int(segment_dictionary["end_time"] * sampling_rate)

        end_sample_index = min(end_sample_index, total_audio_samples)

        sliced_chunk = audio_data[start_sample_index:end_sample_index]

        chunked_audio_dir = os.path.join(
            audio_dir,
            f"{ASR}_chunk_{segment_index:04d}_{segment_dictionary["speaker"]}.wav"
        )
        soundfile.write(chunked_audio_dir, sliced_chunk, sampling_rate)

        chunk_metadata_list.append({
            "file_path": chunked_audio_dir,
            "speaker": segment_dictionary["speaker"]
        })

        print("Chunking complete.")
        print(f"Chunking files are saved in: {chunked_audio_dir}\n")

    return chunk_metadata_list

# transcribs the audio chunks created before
# plus, loads the model into memory
# considering 6gb vram
def transcriber(chunk_metadata_list: list, hf_model: str, device: str) -> list:
    # initialise the speech recognition pipeline
    transcription_pipeline = TRANS_PIPELINE(
        "automatic-speech-recognition",
        model = hf_model,
        device = device,
        torch_dtype = torch.float32
        #torch.float16 if torch.cuda.is_available() else torch.float32
    )

    transcribed_results = []

    for chunk_dict in chunk_metadata_list:
        file_to_process = chunk_dict["file_path"]

        try:
            model_output = transcription_pipeline(
                file_to_process,
                chunk_length_s = 30,
                generate_kwargs = {"language": "bengali", "task": "transcribe"}
            )

            chunk_dict["transcription"] = model_output.get("text", "").strip()
            transcribed_results.append(chunk_dict)

            print("Successfully completed transcription.\n")
            print("Attemping memory clearance...")

        finally:
            if os.path.exists(file_to_process):
                os.remove(file_to_process)
            print("Memory cleared!\n")
    
    return transcribed_results

# save the final transcript
def save_transcript(transcribed_results: list, output_file_name: str) -> None:

    output_path = os.path.join(output_dir, output_file_name)

    with open(output_path, "w", encoding="utf-8") as file:
        for chunk_dict in transcribed_results:
            speaker_label = chunk_dict["speaker"]
            transcription = chunk_dict["transcription"]

            formatted_line = f"[{speaker_label}] {transcription}\n"
            file.write(formatted_line)
    
    print(f"Successfully saved the transcript on: {output_path}")

# trims audio if instructed
def trim_audio(audio_src: str, duration: float) -> str:
    audio_data, sampling_rate = soundfile.read(audio_src)

    target_index = int(duration * sampling_rate)
    target_index = min(target_index, len(audio_data))

    trimmed_audio = audio_data[:target_index]
    trimmed_audio_path = audio_src.replace(".wav", "_trimmed.wav")

    soundfile.write(trimmed_audio_path, trimmed_audio, sampling_rate)

    print(f"\n[SUCCESS] audio successfully trimmed to {duration} seconds.\n")
    return trimmed_audio_path


def main():
    if len(sys.argv) < 2:
        print("\n[ERROR] usage: python tituFC_asr.py <audio_name> <duration (optional)>\n")
        sys.exit(1)

    if os.path.exists(os.path.join(audio_dir, f"{sys.argv[1]}.wav")):
        print("\n[SUCCESS] audio file exists. processing speech to text...\n")
    else:
        print(f"\n[ERROR] audio file not found. please refer to {os.path.join(root, 'download_audio.py')} to download the audio first.\n")
        sys.exit(1)

    print(f"script location: {script_dir}")
    print(f"root asr folder: {root}")
    print(f"audio directory: {audio_dir}")
    print(f"output directory: {output_dir}\n")

    print(f"we're working on: {device}\n")

    audio_name = sys.argv[1]
    audio_path = os.path.join(audio_dir, f"{audio_name}.wav")
    model_id = "hishab/titu_stt_bn_fastconformer"
    output_file_name = f"{ASR}_transcript.txt"

    if sys.argv[2]:
        duration = int(sys.argv[2])
        audio_path = trim_audio(audio_path, duration)

    speakers = speaker_diarisation(audio_path, device)
    print(f"[SUCCESS] found {len(speakers)} speaker segments.\n")

    chunk_metadata = chunking(audio_path, speakers)

    transcription = transcriber(chunk_metadata, model_id, device)

    save_transcript(transcription, output_file_name)

    print("\nSUCCESS!")
    print("COMPLETED TRANSCRIPT GENERATION!\n")

if __name__ == "__main__":
    main()