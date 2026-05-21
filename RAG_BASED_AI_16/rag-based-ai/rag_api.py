# rag_api.py
import os
import time
import joblib
import json
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_BASED_AI_16\rag-based-ai"
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.joblib")
RAW_OLLAMA_LOG = os.path.join(BASE_DIR, "raw_ollama_response.json")

# configuration
TOP_RESULTS = 5
MAX_CONTEXT_CHARS = 3000

# load embeddings once
_df = None
_embeddings_matrix = None

def load_embeddings():
    global _df, _embeddings_matrix
    if _df is None:
        _df = joblib.load(EMBEDDINGS_FILE)
        try:
            _embeddings_matrix = np.vstack(_df["embedding"].values)
        except Exception as e:
            _embeddings_matrix = None
            raise
    return _df, _embeddings_matrix

def create_embedding(text_list):
    """Call Ollama bge-m3 for embeddings (returns list of vectors)."""
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3:latest", "input": text_list},
            timeout=60
        )
        r.raise_for_status()
        return r.json().get("embeddings", [])
    except Exception as e:
        # fallback zeros (same dim as before)
        print(f"[rag_api] embedding error: {e}")
        return [np.zeros(1024).tolist() for _ in text_list]

def call_ollama_generate(prompt, model="llama3:latest"):
    """Call Ollama generate and save raw json for debug."""
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        time.sleep(0.5)
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        raw = r.json()
        try:
            with open(RAW_OLLAMA_LOG, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return raw
    except Exception as e:
        print(f"[rag_api] generate error: {e}")
        return None

def extract_text(raw):
    """Robustly pull text from Ollama response."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        for k in ("response","output","generated_text","result"):
            v = raw.get(k)
            if v:
                if isinstance(v, list):
                    return " ".join(map(str,v))
                return str(v)
        if "choices" in raw and isinstance(raw["choices"], list) and raw["choices"]:
            c0 = raw["choices"][0]
            for k in ("message","text","output","content"):
                if k in c0:
                    return str(c0[k])
    # fallback to stringified
    return json.dumps(raw, ensure_ascii=False)

def build_prompt(query, top_chunks):
    """
    Build an instructor-style prompt: explanation + timestamps.
    top_chunks: DataFrame slice with title, number, start, end, text
    """
    # create short context
    pieces = []
    for _, row in top_chunks.iterrows():
        pieces.append(f"[Video {row.get('number')}] {row.get('title')} ({int(row.get('start',0))}-{int(row.get('end',0))}s): {row.get('text')}")
    context = "\n\n".join(pieces)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rsplit("\n",1)[0]

    prompt = f"""You are an AI Teaching Assistant for a Web Development course.
Use ONLY the context below to answer the user's question. Explain the concept in clear paragraphs and then list the video number/title and timestamp ranges where the topic appears.

Context:
{context}

Question:
{query}

Instructions:
- Explain in clear, structured paragraphs like a real instructor.
- After explaining, list the relevant video number/title and timestamp ranges found in the context.
- If the question is unrelated to this course, reply: "I can only answer questions related to this course."
"""
    return prompt

def answer_query(query, top_k=TOP_RESULTS, model="llama3:latest"):
    """
    Main function to call from UI.
    Returns dict: { 'answer': str, 'sources': [ {title, number, start,end,text} ], 'raw': raw_ollama_json }
    """
    df, emb_mat = load_embeddings()
    # get query embedding
    q_emb = create_embedding([query])[0]
    if emb_mat is None:
        return {"answer": "Embeddings matrix not available.", "sources": [], "raw": None}
    sims = cosine_similarity(emb_mat, [q_emb]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    top_df = df.loc[top_idx].reset_index(drop=True)

    prompt = build_prompt(query, top_df)
    raw = call_ollama_generate(prompt, model=model)
    text = extract_text(raw)
    # prepare sources list
    sources = []
    for _, row in top_df.iterrows():
        sources.append({
            "title": row.get("title"),
            "number": row.get("number"),
            "start": int(row.get("start",0)),
            "end": int(row.get("end",0)),
            "text": row.get("text")
        })
    return {"answer": text, "sources": sources, "raw": raw}
