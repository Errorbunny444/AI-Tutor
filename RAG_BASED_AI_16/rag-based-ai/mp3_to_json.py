import whisper
import json
import os

# === CONFIG ===
AUDIO_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_ALL_AUDIOS"
JSON_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_JSONS"

# Make sure output folder exists
os.makedirs(JSON_DIR, exist_ok=True)

# Load Whisper model (use large-v2 for max accuracy)
print("🔊 Loading Whisper model 'large-v2' ...")
model = whisper.load_model("large-v2")
print("✅ Model loaded successfully!\n")

# List all audio files
audios = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".mp3")]

print(f"🎧 Found {len(audios)} audio files to process.\n")

for audio in audios:
    try:
        if "_" in audio:
            number = audio.split("_")[0]
            title = audio.split("_", 1)[1][:-4]

            input_path = os.path.join(AUDIO_DIR, audio)
            output_path = os.path.join(JSON_DIR, f"{audio}.json")

            print(f"▶️ Transcribing: {audio} ...")

            # Run transcription
            result = model.transcribe(
                audio=input_path,
                language="hi",        # change to "en" if your videos are in English
                task="translate",     # "transcribe" if you want same language
                word_timestamps=False
            )

            # Create structured chunks
            chunks = [
                {
                    "number": number,
                    "title": title,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in result.get("segments", [])
            ]

            chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

            # Save transcript to JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

            print(f"✅ Saved: {output_path}\n")

    except Exception as e:
        print(f"❌ Error processing {audio}: {e}\n")
        continue

print("🏁 All audio files have been processed and saved as JSONs successfully!")
