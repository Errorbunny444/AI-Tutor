import os
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import json

# === CONFIGURATION ===
BASE_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_BASED_AI_16\rag-based-ai"
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.joblib")
PROMPT_LOG = os.path.join(BASE_DIR, "prompt.txt")
RESPONSE_LOG = os.path.join(BASE_DIR, "response.txt")
RAW_OLLAMA_LOG = os.path.join(BASE_DIR, "raw_ollama_response.json")
TOP_RESULTS = 5
MAX_CONTEXT_CHARS = 3000  # To prevent huge prompts

# === UTILITY: OLLAMA EMBEDDING ===
def create_embedding(text_list):
    """Generate embeddings using Ollama’s bge-m3 model."""
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3:latest", "input": text_list},
            timeout=60
        )
        r.raise_for_status()
        embedding = r.json().get("embeddings", [])
        return embedding
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return [np.zeros(1024).tolist() for _ in text_list]

# === UTILITY: OLLAMA INFERENCE (Improved) ===
def call_ollama_generate(prompt):
    """Call Ollama generate endpoint and return parsed JSON (and save raw json)."""
    try:
        payload = {
            "model": "llama3:latest",
            "prompt": prompt,      # ✅ switched back from 'input' to 'prompt'
            "stream": False
        }

        time.sleep(1)  # ensures model warm-up on first call
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        raw = r.json()

        with open(RAW_OLLAMA_LOG, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, ensure_ascii=False, indent=2)

        return raw
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None

def extract_text_from_ollama_response(raw):
    """Extract text from Ollama’s response safely."""
    if raw is None:
        return "⚠️ No raw response from Ollama.", "no_raw"

    if isinstance(raw, dict):
        for key in ["response", "output", "generated_text", "result"]:
            val = raw.get(key)
            if val:
                if isinstance(val, list):
                    return " ".join(map(str, val)), key
                return str(val), key
        if "choices" in raw and isinstance(raw["choices"], list) and raw["choices"]:
            c0 = raw["choices"][0]
            for k in ("message", "text", "output", "content"):
                if k in c0:
                    return str(c0[k]), f"choices[0].{k}"
    return json.dumps(raw, ensure_ascii=False), "stringified"

# === LOAD EMBEDDINGS ===
print("📦 Loading embeddings...")
df = joblib.load(EMBEDDINGS_FILE)
print(f"✅ Loaded {len(df)} text chunks.\n")

# === MAIN LOOP ===
while True:
    incoming_query = input("🎓 Ask a Question (or type 'exit' to quit): ").strip()
    if incoming_query.lower() in ["exit", "quit"]:
        print("👋 Goodbye, happy learning!")
        break

    print("🔍 Searching for relevant content...\n")
    question_embedding = create_embedding([incoming_query])[0]

    try:
        embeddings_matrix = np.vstack(df["embedding"].values)
    except Exception as e:
        print(f"❌ Failed to build embeddings matrix: {e}")
        continue

    similarities = cosine_similarity(embeddings_matrix, [question_embedding]).flatten()
    max_indx = similarities.argsort()[::-1][:TOP_RESULTS]
    new_df = df.loc[max_indx]

    # Build SHORT context
    chunk_texts = []
    for _, row in new_df.iterrows():
        piece = (
            f"[Video {row.get('number')}] {row.get('title')} "
            f"({int(row.get('start',0))}-{int(row.get('end',0))}s): {row.get('text')}"
        )
        chunk_texts.append(piece)

    context_joined = "\n\n".join(chunk_texts)
    if len(context_joined) > MAX_CONTEXT_CHARS:
        context_joined = context_joined[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0]

    # === UPDATED PROMPT ===
    prompt = f"""
You are an **AI Teaching Assistant** for a *Web Development* course.
Your job is to help students by explaining topics from the course using the context below.

Context:
{context_joined}

Question:
{incoming_query}

Instructions:
- Explain the concept in clear, structured paragraphs like a real instructor would.
- Use the transcript context to build an accurate, helpful explanation.
- After explaining, list the relevant **video number, title, and timestamp ranges** where this concept appears.
- Keep the tone friendly, educational, and easy to understand.
- Format your final answer neatly with line breaks and bullet points if needed.
- If the question is unrelated to this course, reply: "I can only answer questions related to this course."
"""

    os.makedirs(os.path.dirname(PROMPT_LOG), exist_ok=True)
    with open(PROMPT_LOG, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("🤖 Thinking...\n")
    raw = call_ollama_generate(prompt)
    text, method = extract_text_from_ollama_response(raw)

    if not text or text.strip() == "" or text.strip().startswith("{"):
        debug_msg = f"(Empty/unknown response extracted via {method})"
        print("⚠️ Model returned empty or unknown response. See raw log for details.")
        text_to_show = f"{debug_msg}\n\nRaw response saved to: {RAW_OLLAMA_LOG}"
    else:
        text_to_show = text.strip()

    print(f"💬 Assistant (extracted via {method}):\n\n{text_to_show}\n")
    with open(RESPONSE_LOG, "w", encoding="utf-8") as f:
        f.write(text_to_show)
