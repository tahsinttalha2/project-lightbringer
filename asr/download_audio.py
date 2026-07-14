import os
import yt_dlp
import json
import sys

root = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(root, "audio")
output_dir = os.path.join(root, "output")

os.makedirs(audio_dir, exist_ok = True)
os.makedirs(output_dir, exist_ok = True)

# extracts youtube metadata and downloads audio
def extract_yt_data(url: str) -> None:

    AUDIO_OUTPUT = os.path.join(audio_dir, "audio")

    # set the download rules
    download_options = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "outtmpl": AUDIO_OUTPUT,
        "getcomments": True,
        "quiet": False,
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": ["bn", "bn-in", "bn-bd", "bengali", "bangla"],
        "subtitlesformat": "vtt"
    }

    # instatiate the download
    with yt_dlp.YoutubeDL(download_options) as downloader:
        video_information = downloader.extract_info(url, download = True) 

        available_manual_subtitles = video_information.get("subtitles", {})
        has_bengali_transcript = "bn" in available_manual_subtitles

        extracted_metadata = {
            "video_id": video_information.get("id"),
            "title": video_information.get("title"),
            "description": video_information.get("description"),
            "upload_date": video_information.get("upload_date"),
            "like_count": video_information.get("like_count"),
            "channel": video_information.get("channel"),
            "subscriber_count": video_information.get("channel_follower_count"),
            "view_count": video_information.get("view_count"),
            "categories": video_information.get("categories"),
            "tags": video_information.get("tags"),
            "has_manual_bengali_transcript": has_bengali_transcript,
            "comments": []
        }

        # iterate through comments if they're enabled
        all_comments = video_information.get("comments")

        if all_comments:
            for comment in all_comments:
                extracted_metadata["comments"].append({
                    "author": comment.get("author"),
                    "text": comment.get("text"),
                    "like_count": comment.get("like_count"),
                    "loved": comment.get("is_favorited")
                })

        OUTPUT = os.path.join(output_dir, f"{extracted_metadata["video_id"]}_metadata.json")

        with open(OUTPUT, "w", encoding = "utf-8") as file:
            json.dump(extracted_metadata, file, ensure_ascii = False, indent = 4)

        print(f"\naudio saved in: {AUDIO_OUTPUT}")
        print(f"file saved in: {OUTPUT}\n")

def main():
    if len(sys.argv) < 2:
        print("\n[ERROR] usage: python download_audio.py <youtube url>\n")
        sys.exit(1)

    url = sys.argv[1]
    extract_yt_data(url)

if __name__ == "__main__":
    main()