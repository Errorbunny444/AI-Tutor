import os
import json
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIGURATION ===
JSON_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_JSONS"
OUTPUT_FILE = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_BASED_AI_16\rag-based-ai\embeddings.joblib"

# === CREATE EMBEDDING FUNCTION ===
def create_embedding(text_list):
    """
    Calls Ollama’s local API to create embeddings.
    Ensure Ollama is running and model bge-m3 is loaded:
        ollama run bge-m3
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text_list},
            timeout=180
        )
        r.raise_for_status()
        return r.json().get("embeddings", [])
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        # return placeholder zero-vectors if embedding call fails
        return [np.zeros(1024).tolist() for _ in text_list]

# === STEP 1: COLLECT & MERGE JSON FILES ===
print("🔍 Scanning JSON files for preprocessing...")
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
print(f"✅ Found {len(json_files)} JSON files.\n")

merged_chunks = []
chunk_id = 0

for json_file in json_files:
    file_path = os.path.join(JSON_DIR, json_file)
    print(f"📂 Reading {json_file} ...")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        text_segments = [c["text"] for c in content.get("chunks", [])]

        # === STEP 2: CREATE EMBEDDINGS FOR THIS FILE ===
        print(f"🎯 Creating embeddings for {len(text_segments)} chunks ...")
        embeddings = create_embedding(text_segments)

        # === STEP 3: ATTACH METADATA ===
        for i, chunk in enumerate(content.get("chunks", [])):
            chunk["chunk_id"] = chunk_id
            chunk["source_file"] = json_file
            chunk["embedding"] = embeddings[i] if i < len(embeddings) else np.zeros(1024).tolist()
            chunk_id += 1
            merged_chunks.append(chunk)

    except Exception as e:
        print(f"⚠️  Skipping {json_file}: {e}")

# === STEP 4: SAVE MERGED DATA ===
print("\n💾 Saving merged embeddings dataframe...")
df = pd.DataFrame.from_records(merged_chunks)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
joblib.dump(df, OUTPUT_FILE)

print("\n✅ Preprocessing & embedding creation complete!")
print(f"📦 {len(df)} text chunks saved with embeddings at:")
print(f"👉 {OUTPUT_FILE}")
