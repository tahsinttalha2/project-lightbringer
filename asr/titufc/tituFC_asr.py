import os

import torch
#from pyannote.audio import Pipeline as PIPELINE
import gc

import soundfile
import nemo.collections.asr as NEMO

import sys

import warnings
warnings.filterwarnings("ignore")

ASR = "tituFC"

# dynamic directory paths for consistent file operation
script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(script_dir)
audio_dir = os.path.join(root, "audio")
output_dir = os.path.join(root, "output")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transcriber(audio_src: list, hf_model: str, device: str) -> list:
    # initialise the speech recognition pipeline
    model = NEMO.models.ASRModel.from_pretrained(model_name = hf_model)
    model.change_attention_model(self_attention_model = "rel_pos_local_attn", att_context_size = [128, 128])
    model.change_subsampling_conv_chunking_factor(1)

    model.cfg.decoding.strategy = "greedy_batch"
    model.change_decoding_strategy(model.cfg.decoding)

    model = model.to(device)
 
    output = model.transcribe([audio_src])

    if isinstance(output, tuple):
        extracted_data = output[0][0]
    else:
        extracted_data = output[0]

    # extracts the text from extracted data
    if hasattr(extracted_data, "text"):
        extracted_text = extracted_data.text
    else:
        extracted_text = str(extracted_data)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Successfully completed transcription.\n")
    print("Attemping memory clearance...")
    
    del model
    gc.collect()

    print("[DONE]\n")

    return extracted_text.strip()

# save the final transcript
def save_transcript(transcription: str, output_file_name: str) -> None:

    output_path = os.path.join(output_dir, output_file_name)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(transcription)
    
    print(f"Successfully saved the transcript on: {output_path}")

# trims audio if instructed
def prepare_audio(audio_src: str, duration: float = None) -> str:
    audio_data, sampling_rate = soundfile.read(audio_src)

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis = 1)

    if duration is not None:
        target_index = int(duration * sampling_rate)
        target_index = min(target_index, len(audio_data))
        audio_data = audio_data[:target_index]

    processed_audio = audio_src.replace(".wav", "_processed.wav")

    soundfile.write(processed_audio, audio_data, sampling_rate)

    print(f"\n[SUCCESS] audio successfully trimmed to {duration} seconds.\n")
    return processed_audio


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

    if len(sys.argv) > 2:
        duration = float(sys.argv[2])
        audio_path = prepare_audio(audio_path, duration)
    else:
        audio_path = prepare_audio(audio_path, None)

    transcription = transcriber(audio_path, model_id, device)

    save_transcript(transcription, output_file_name)

    print("\nSUCCESS!")
    print("COMPLETED TRANSCRIPT GENERATION!\n")

if __name__ == "__main__":
    main()