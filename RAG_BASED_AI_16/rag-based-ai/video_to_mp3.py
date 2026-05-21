import os
import subprocess

# === Define your input/output folders ===
VIDEO_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_SAMPLE_VIDEOS"
AUDIO_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_ALL_AUDIOS"

# === Create the output folder if not exists ===
os.makedirs(AUDIO_DIR, exist_ok=True)

# === Loop through each video file ===
for file in os.listdir(VIDEO_DIR):
    if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue  # skip non-video files

    input_path = os.path.join(VIDEO_DIR, file)
    file_name = os.path.splitext(file)[0]
    output_path = os.path.join(AUDIO_DIR, f"{file_name}.mp3")

    print(f"🎬 Converting: {file} → {output_path}")
    subprocess.run(["ffmpeg", "-i", input_path, "-vn", "-acodec", "mp3", output_path])

print("\n✅ All videos have been processed successfully!")
