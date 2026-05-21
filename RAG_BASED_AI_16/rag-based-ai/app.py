# app.py
"""
Fully upgraded RAG-based AI Teaching Assistant
Includes:
 - Whisper voice input
 - Edge-TTS voice output
 - Local Ollama embeddings + generation (assumes Ollama HTTP API running locally)
 - Knowledge uploads (PDF/TXT/DOCX)
 - RAG retrieval + inline citation labeling (KB# / WEB#)
 - Lightweight live web snippets (DuckDuckGo Instant Answer) for citations
 - Quiz generation & grading (embedding similarity)
 - Context-aware follow-up suggestions (Step 11)
 - Session-state safe usage and unique widget keys to avoid duplicate element errors
 - PDF export and audio downloads per chat item
Note: Adjust BASE_DIR, local endpoints and installed packages as needed.
"""

import os
import time
import json
import uuid
import re
import tempfile
import random
import asyncio
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from fpdf import FPDF
import edge_tts
from PyPDF2 import PdfReader
from docx import Document

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = r"E:\SY\DS\RAG_BASED_AI_PROJECT\RAG_BASED_AI_16\rag-based-ai"
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.joblib")
RAW_OLLAMA_LOG = os.path.join(BASE_DIR, "raw_ollama_response.json")
TOP_RESULTS = 6
MAX_CONTEXT_CHARS = 3500
VOICE_DURATION = 10  # seconds
MEMORY_LIMIT = 3  # previous turns to remember
ENABLE_LIVE_SEARCH_BY_DEFAULT = True

# Make sure base dir exists (safe-guard)
os.makedirs(BASE_DIR, exist_ok=True)

# ---------------------------
# CACHED RESOURCE LOADERS
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource(show_spinner=False)
def load_embeddings() -> pd.DataFrame:
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            return joblib.load(EMBEDDINGS_FILE)
        except Exception:
            # corrupted - recreate
            return pd.DataFrame(columns=["number", "title", "start", "end", "text", "embedding", "source"])
    else:
        return pd.DataFrame(columns=["number", "title", "start", "end", "text", "embedding", "source"])

# Load resources
df = load_embeddings()
model_whisper = load_whisper_model()

# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🎙️ Adaptive RAG AI Teaching Assistant + Follow-ups", layout="wide")

# initialize session state keys safely
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []          # list of {"user","assistant","citations","followups"}
if "voice_query" not in st.session_state:
    st.session_state["voice_query"] = ""
if "current_quiz" not in st.session_state:
    st.session_state["current_quiz"] = []
if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = []
if "quiz_results" not in st.session_state:
    st.session_state["quiz_results"] = None
if "enable_live_search" not in st.session_state:
    st.session_state["enable_live_search"] = ENABLE_LIVE_SEARCH_BY_DEFAULT
if "last_action_id" not in st.session_state:
    st.session_state["last_action_id"] = str(int(time.time()))

# ---------------------------
# UTIL: Embeddings & Ollama
# ---------------------------
def create_embedding(text_list: List[str]) -> List[List[float]]:
    """
    POST to local Ollama embeddings endpoint:
    http://localhost:11434/api/embed
    Model: bge-m3:latest
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3:latest", "input": text_list},
            timeout=60
        )
        r.raise_for_status()
        return r.json().get("embeddings", [])
    except Exception as e:
        st.warning(f"Embedding generation failed: {e}")
        # fallback to zero vectors sized 1024
        return [np.zeros(1024).tolist() for _ in text_list]

def call_ollama_generate(prompt: str, model_name: str = "llama3:latest", timeout: int = 180) -> str:
    """
    Call local Ollama generate endpoint and return string output.
    """
    try:
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json()
        # save raw for debugging
        try:
            with open(RAW_OLLAMA_LOG, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if isinstance(raw, dict):
            return raw.get("response", "") or raw.get("output", "") or json.dumps(raw, ensure_ascii=False)
        return str(raw)
    except Exception as e:
        return f"⚠️ Ollama failed: {e}"

# ---------------------------
# UTIL: Live Web Snippet (DuckDuckGo Instant Answer)
# ---------------------------
def live_search_duckduckgo(query: str, max_results: int = 3) -> List[Dict]:
    try:
        params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
        r = requests.get("https://api.duckduckgo.com/", params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        results = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading") or "DuckDuckGo Abstract",
                "snippet": data.get("AbstractText"),
                "url": data.get("AbstractURL") or "https://duckduckgo.com/"
            })
        for rt in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(rt, dict):
                txt = rt.get("Text", "")
                first_url = rt.get("FirstURL", "")
                results.append({"title": rt.get("Name") or txt[:60], "snippet": txt, "url": first_url})
        return results[:max_results]
    except Exception:
        return []

# ---------------------------
# AUDIO: Recording & Whisper
# ---------------------------
def record_voice(duration: int = VOICE_DURATION) -> str:
    st.info("🎙️ Recording... Speak now!")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("✅ Recording complete!")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp.name, fs, audio)
    return tmp.name

def transcribe_audio(audio_path: str) -> str:
    try:
        result = model_whisper.transcribe(audio_path, language="en")
        return result.get("text", "")
    except Exception as e:
        st.warning(f"Whisper transcription failed: {e}")
        return ""

# ---------------------------
# PDF Export
# ---------------------------
def export_to_pdf(title_text: str, body_text: str) -> str:
    def clean_for_pdf(text: str) -> str:
        text = text.replace("–", "-").replace("—", "-")
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("•", "-")
        return text.encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title_text, ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, clean_for_pdf(body_text))
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(out.name, "F")
    return out.name

# ---------------------------
# Edge TTS helpers
# ---------------------------
async def synthesize_voice_async(text: str, voice: str) -> str:
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts = edge_tts.Communicate(text, voice)
    await tts.save(out_file.name)
    return out_file.name

def text_to_speech_sync(text: str, voice: str) -> Optional[str]:
    try:
        return asyncio.run(synthesize_voice_async(text, voice))
    except Exception as e:
        st.warning(f"TTS Error: {e}")
        return None

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?s\)", "", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"#\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# ---------------------------
# RAG helpers: retrieval, citations, prompt composition
# ---------------------------
def find_top_k_chunks(query: str, top_k: int = TOP_RESULTS) -> Tuple[pd.DataFrame, List[float]]:
    if len(df) == 0:
        return pd.DataFrame(), []
    try:
        embeddings_matrix = np.vstack(df["embedding"].values)
    except Exception:
        return df.head(top_k), [0.0] * min(top_k, len(df))
    q_emb = create_embedding([query])[0]
    sims = cosine_similarity(embeddings_matrix, [q_emb]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    top_df = df.iloc[top_idx]
    top_scores = sims[top_idx].tolist()
    return top_df, top_scores

def prepare_citation_list(top_df: pd.DataFrame, top_scores: List[float], live_snippets: List[Dict]) -> List[Dict]:
    citations = []
    for i, (idx, row) in enumerate(top_df.iterrows()):
        label = f"KB#{int(row.get('number', idx))}" if row.get("number", None) is not None else f"KB#{idx}"
        snippet = str(row.get("text", ""))[:280].strip().replace("\n", " ")
        src = row.get("source", "")
        citations.append({
            "id": label,
            "label": label,
            "source": src or "Local KB",
            "score": float(top_scores[i]) if i < len(top_scores) else 0.0,
            "snippet": snippet,
            "url": src or ""
        })
    for i, item in enumerate(live_snippets):
        label = f"WEB#{i+1}"
        citations.append({
            "id": label,
            "label": label,
            "source": item.get("title", "Web"),
            "score": 0.0,
            "snippet": item.get("snippet", "")[:320],
            "url": item.get("url", "")
        })
    return citations

def generate_response_with_sources(query: str, personality: str, enable_live_search: bool) -> Tuple[str, List[Dict]]:
    top_df, top_scores = find_top_k_chunks(query, TOP_RESULTS)
    context_chunks = []
    for _, row in top_df.iterrows():
        num = row.get("number", "")
        title = row.get("title", "") or ""
        start = int(row.get("start", 0)) if row.get("start", 0) != "" else 0
        end = int(row.get("end", 0)) if row.get("end", 0) != "" else 0
        text = row.get("text", "")
        context_chunks.append(f"[Video {num}] {title} ({start}-{end}s): {text}")
    rag_context = "\n\n".join(context_chunks)[:MAX_CONTEXT_CHARS]

    memory_context = ""
    if st.session_state["chat_history"]:
        recent = st.session_state["chat_history"][-MEMORY_LIMIT:]
        memory_context = "\n\n".join([f"User: {t['user']}\nAssistant: {t['assistant']}" for t in recent])

    live_snippets = []
    if enable_live_search:
        try:
            live_snippets = live_search_duckduckgo(query, max_results=3)
        except Exception:
            live_snippets = []

    citations = prepare_citation_list(top_df, top_scores, live_snippets)

    if "Professor" in personality:
        tone = "You are a formal and precise web development professor who explains with academic clarity."
    elif "Tech Bro" in personality or "Tech" in personality:
        tone = "You are a casual tech-bro tutor — confident, concise, and full of practical analogies."
    else:
        tone = "You are a friendly, encouraging mentor. Use examples and motivation."

    prompt = f"""
You are an advanced AI teaching assistant. Follow these instructions.

Tone:
{tone}

Conversation Memory:
{memory_context}

Relevant Local Knowledge (KB chunks below):
{rag_context}

Live Web Snippets (as JSON):
{json.dumps(live_snippets, ensure_ascii=False) if live_snippets else ''}

Citations:
When using information from the Knowledge Base or Live Web Snippets, include inline citations like (KB#12) or (WEB#1).
At the end include a short 'References' list mapping cited labels to short snippets/URLs.

User Question:
{query}

Instructions:
- Use only the provided context + live snippets for factual claims.
- Include inline citations for claims derived from KB or web (e.g., (KB#5), (WEB#2)).
- Structure: 1) Overview 2) Detailed explanation 3) References (list cited labels + one-line snippet) 4) Summary.
- If unsupported by context, be honest and suggest follow-up questions.
"""
    assistant_text = call_ollama_generate(prompt)
    return assistant_text, citations

# ---------------------------
# QUIZ / EVAL helpers
# ---------------------------
def generate_quiz_from_context(context_text: str, num_questions: int = 5) -> List[Dict]:
    prompt = f"""
You are an instructor creating short-answer quiz questions from the context below.
Return exactly a JSON array of objects: [{{"q":"...","a":"..."}}, ...]
Context:
{context_text}

Produce {num_questions} Q/A pairs. Keep answers concise (1-2 sentences).
"""
    raw = call_ollama_generate(prompt)
    try:
        start = raw.find("[")
        if start != -1:
            json_part = raw[start:]
            quiz = json.loads(json_part)
            cleaned = []
            for item in quiz:
                if isinstance(item, dict) and "q" in item and "a" in item:
                    cleaned.append({"q": item["q"].strip(), "a": item["a"].strip()})
            if cleaned:
                return cleaned
    except Exception:
        pass

    # fallback parse
    qa_pairs = []
    lines = raw.splitlines()
    q, a = None, None
    for line in lines:
        line = line.strip()
        if line.lower().startswith("q:") or line.lower().startswith("question"):
            q = re.sub(r'^(q:|question:)\s*', '', line, flags=re.I)
            a = None
        elif line.lower().startswith("a:") or line.lower().startswith("answer"):
            a = re.sub(r'^(a:|answer:)\s*', '', line, flags=re.I)
        else:
            if q and not a:
                q += " " + line
            elif a:
                a += " " + line
        if q and a:
            qa_pairs.append({"q": q.strip(), "a": a.strip()})
            q, a = None, None
        if len(qa_pairs) >= num_questions:
            break
    return qa_pairs

def grade_answer(expected_answer: str, user_answer: str) -> Tuple[float, float]:
    try:
        emb = create_embedding([expected_answer, user_answer])
        e_emb = np.array(emb[0])
        u_emb = np.array(emb[1])
        sim = float(cosine_similarity([e_emb], [u_emb])[0][0])
        score = float(max(0.0, min(1.0, (sim - 0.45) / (0.35))))
        score = max(0.0, min(1.0, score))
        return score, sim
    except Exception:
        return 0.0, 0.0

# ---------------------------
# FOLLOW-UP SUGGESTIONS (Step 11)
# ---------------------------
def generate_followups(answer_text: str, user_query: str, max_suggestions: int = 4) -> List[str]:
    """
    Generate context-aware follow-up question suggestions based on assistant's answer and original query.
    We'll ask Ollama to produce short follow-up questions in JSON.
    """
    prompt = f"""
You are a helpful tutor assistant. Given the original user question and the assistant's answer, propose up to {max_suggestions} concise follow-up questions or study tasks the student could ask next. Return a JSON array of short strings.

Original Question:
{user_query}

Assistant's Answer:
{answer_text}

Return JSON like:
["Follow-up question 1", "Follow-up question 2", ...]
"""
    raw = call_ollama_generate(prompt, timeout=40)
    try:
        start = raw.find("[")
        if start != -1:
            arr = json.loads(raw[start:])
            if isinstance(arr, list):
                cleaned = [str(x).strip() for x in arr][:max_suggestions]
                return cleaned
    except Exception:
        pass

    # fallback: extract short sentences/questions
    suggestions = []
    lines = raw.splitlines()
    for line in lines:
        line = line.strip().strip("-*• ")
        if line and len(suggestions) < max_suggestions:
            suggestions.append(line)
    # if none, produce generic ones
    if not suggestions:
        suggestions = [
            "Can you provide a short example that applies this?",
            "What are common mistakes to avoid here?",
            "How does this connect to X topic?",
            "Give a step-by-step solution for a simple case."
        ][:max_suggestions]
    return suggestions

# ============================================================
# STEP 12 — PERFORMANCE ANALYTICS MODE
# ============================================================
if "metrics" not in st.session_state:
    st.session_state["metrics"] = {
        "total_turns": 0,
        "avg_response_time": [],
        "avg_turn_length": [],
        "citations_used": 0,
        "followups_used": 0,
        "quiz_scores": []
    }

# --- instrumentation helper ---
def log_performance_metrics(start_t: float, assistant_text: str, citations: List[Dict], followups: List[str]):
    elapsed = time.time() - start_t
    st.session_state["metrics"]["total_turns"] += 1
    st.session_state["metrics"]["avg_response_time"].append(elapsed)
    st.session_state["metrics"]["avg_turn_length"].append(len(assistant_text))
    if citations:
        st.session_state["metrics"]["citations_used"] += 1
    if followups:
        st.session_state["metrics"]["followups_used"] += 1

# --- integrate with answer generation ---
# inside the "Ask (Teaching Mode)" button block, just after you get assistant_text:
# (look for line: assistant_text, citations = generate_response_with_sources(...))
# ADD:
# start_t = time.time()
# assistant_text, citations = generate_response_with_sources(...)
# followups = generate_followups(...)
# log_performance_metrics(start_t, assistant_text, citations, followups)

# --- integrate quiz scores after grading ---
# After grading quiz (where total_score is computed):
# st.session_state["metrics"]["quiz_scores"].append(total_score)

# --- new Dashboard tab ---
with st.sidebar.expander("📊 Performance Dashboard", expanded=False):
    m = st.session_state["metrics"]
    if m["total_turns"] > 0:
        avg_resp = np.mean(m["avg_response_time"])
        avg_len = np.mean(m["avg_turn_length"])
        cite_rate = (m["citations_used"] / m["total_turns"]) * 100
        fu_rate = (m["followups_used"] / m["total_turns"]) * 100
        avg_quiz = np.mean(m["quiz_scores"]) if m["quiz_scores"] else 0

        st.metric("Total Turns", m["total_turns"])
        st.metric("Avg Response Time (s)", round(avg_resp, 2))
        st.metric("Avg Answer Length (chars)", int(avg_len))
        st.metric("Citation Usage Rate", f"{cite_rate:.1f}%")
        st.metric("Follow-up Suggestion Rate", f"{fu_rate:.1f}%")
        st.metric("Avg Quiz Score", f"{avg_quiz:.1f}%")

        # Small bar chart for visual engagement
        st.markdown("**Session Overview**")
        df_metrics = pd.DataFrame({
            "Metric": ["Resp Time (s)", "Ans Length", "Cite Rate %", "Follow-up Rate %"],
            "Value": [avg_resp, avg_len, cite_rate, fu_rate]
        })
        st.bar_chart(df_metrics.set_index("Metric"))

        st.caption("📈 Metrics refresh each session. Clears when you restart or clear chat.")
    else:
        st.info("No analytics yet — start chatting or take a quiz!")


# ---------------------------
# PAGE STYLING
# ---------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 19px !important; line-height: 1.6; }
    .chat-bubble { background-color: #0f1720; padding: 14px; border-radius: 12px; margin: 10px 0; color: #e6eef6; }
    .user-bubble { border-left: 6px solid #6c63ff; }
    .assistant-bubble { border-left: 6px solid #00bfa6; }
    section[data-testid="stSidebar"] { width: 420px !important; min-width: 420px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# SIDEBAR: Knowledge + Settings
# ---------------------------
with st.sidebar:
    st.header("📘 Knowledge & Settings")
    st.markdown(f"Knowledge entries: **{len(df)}**")
    st.markdown("---")

    mode = st.radio("Mode", ["Teaching Mode", "Evaluation Mode"], index=0)

    st.subheader("Live Search (for citations)")
    live_toggle = st.checkbox("Enable live web snippets (DuckDuckGo)", value=st.session_state["enable_live_search"])
    st.session_state["enable_live_search"] = live_toggle

    st.subheader("Upload / Expand Knowledge")
    uploaded_files = st.file_uploader("Upload PDFs / TXT / DOCX (optional):", accept_multiple_files=True)
    if uploaded_files:
        new_texts = []
        for file in uploaded_files:
            ext = os.path.splitext(file.name)[1].lower()
            text = ""
            if ext == ".pdf":
                try:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                except Exception as e:
                    st.warning(f"PDF read error for {file.name}: {e}")
            elif ext == ".txt":
                try:
                    text = file.read().decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
            elif ext == ".docx":
                try:
                    doc = Document(file)
                    text = "\n".join([p.text for p in doc.paragraphs])
                except Exception as e:
                    st.warning(f"DOCX read error for {file.name}: {e}")
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
            if text.strip():
                chunks = [text[i:i + 800] for i in range(0, len(text), 800)]
                new_texts.extend(chunks)

        if new_texts:
            st.info("Generating embeddings for uploaded knowledge (this may take a few seconds)...")
            embeddings = []
            for i in range(0, len(new_texts), 10):
                batch = new_texts[i:i + 10]
                embeddings.extend(create_embedding(batch))
            new_df = pd.DataFrame({
                "number": range(len(df), len(df) + len(new_texts)),
                "title": ["User Upload"] * len(new_texts),
                "start": [0] * len(new_texts),
                "end": [0] * len(new_texts),
                "text": new_texts,
                "embedding": embeddings,
                "source": [file.name for _ in new_texts]
            })
            df = pd.concat([df, new_df], ignore_index=True)
            joblib.dump(df, EMBEDDINGS_FILE)
            st.success(f"Added {len(new_texts)} new knowledge chunks.")

    if st.button("Clear knowledge base"):
        df = pd.DataFrame(columns=["number", "title", "start", "end", "text", "embedding", "source"])
        joblib.dump(df, EMBEDDINGS_FILE)
        st.warning("Knowledge base cleared!")

    st.markdown("---")
    st.subheader("Tutor Personality")
    personality = st.selectbox("Teaching style:", ["🎓 Formal Professor", "🤝 Friendly Mentor", "😎 Tech Bro Mode"], index=1)

    st.subheader("Voice (Edge-TTS)")
    voice_choice = st.selectbox("Voice:", ["en-US-AriaNeural", "en-GB-RyanNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"], index=0)

    st.caption("Developed by Aryan Ranadive — Whisper + Llama3 + Edge-TTS + RAG + Follow-ups")

# ---------------------------
# MAIN UI: Query area
# ---------------------------
st.title("🎙️ Adaptive Talking RAG-based AI Tutor — Live Sources & Follow-ups")
st.caption("Teaching Mode = chat & explain. Evaluation Mode = generate & grade quizzes.")
st.divider()

st.subheader("Ask / Interact")
col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
query_input = st.text_input("Type your question (or quiz commands):", value=st.session_state.get("voice_query", ""))

# Voice record
with col1:
    if st.button("🎤 Record Voice", key="btn_record"):
        try:
            audio_path = record_voice()
            text = transcribe_audio(audio_path)
            st.session_state["voice_query"] = text
            st.success(f"You said: {text}")
            # don't force rerun; UI will reflect next user action
        except Exception as e:
            st.error(f"Recording/transcription failed: {e}")

# Ask button for Teaching Mode
with col2:
    if st.button("💡 Ask (Teaching Mode)", key="btn_ask"):
        if mode != "Teaching Mode":
            st.info("Switch to Teaching Mode to ask general questions.")
        elif query_input.strip():
            with st.spinner("Thinking..."):
                start_t = time.time()  # start timer
                assistant_text, citations = generate_response_with_sources(
                    query_input, personality=personality, enable_live_search=st.session_state["enable_live_search"]
                )
                
                # optional followups (only if Step 11 was integrated)
                followups = generate_followups(assistant_text, query_input, max_suggestions=4)
                
                # Log metrics (Step 12)
                log_performance_metrics(start_t, assistant_text, citations, followups)
                
                # save chat
                st.session_state["chat_history"].append({
                    "user": query_input,
                    "assistant": assistant_text,
                    "citations": citations,
                    "followups": followups
                })
            
            st.session_state["voice_query"] = ""
            st.rerun()


# Clear chat/memory
with col3:
    if st.button("🧹 Clear Chat & Memory", key="btn_clear"):
        st.session_state["chat_history"] = []
        st.session_state["voice_query"] = ""
        st.success("Cleared chat & memory.")

# ---------------------------
# Evaluation Mode UI
# ---------------------------
if mode == "Evaluation Mode":
    st.markdown("## 🧪 Evaluation Mode (Quiz & Grading)")
    st.info("Generate a quiz from the knowledge base (or uploaded documents).")
    cols = st.columns([0.2, 0.2, 0.6])
    with cols[0]:
        q_count = st.number_input("Questions", min_value=1, max_value=10, value=5, key="q_count")
    with cols[1]:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1, key="difficulty")
    with cols[2]:
        if st.button("Generate Quiz", key="btn_gen_quiz"):
            if len(df) == 0:
                st.warning("Knowledge base empty — add documents first.")
            else:
                sample_df = df.sample(min(len(df), TOP_RESULTS * 3))
                context_text = "\n\n".join(sample_df["text"].tolist())[:MAX_CONTEXT_CHARS]
                with st.spinner("Generating quiz..."):
                    quiz_items = generate_quiz_from_context(context_text, num_questions=q_count)
                    if not quiz_items:
                        st.error("Could not generate quiz. Try again or add more knowledge.")
                    else:
                        st.session_state["current_quiz"] = quiz_items
                        st.session_state["quiz_answers"] = ["" for _ in quiz_items]
                        st.session_state["quiz_results"] = None
                        st.success(f"Quiz with {len(quiz_items)} questions generated.")

    # show quiz if present
    if st.session_state.get("current_quiz"):
        st.markdown("### Quiz")
        quiz = st.session_state["current_quiz"]
        for idx, item in enumerate(quiz):
            st.markdown(f"**Q{idx+1}. {item['q']}**")
            ans = st.text_area(f"Your answer (Q{idx+1})", value=st.session_state["quiz_answers"][idx], key=f"ans_{idx}", height=100)
            st.session_state["quiz_answers"][idx] = ans

        if st.button("Submit Quiz", key="btn_submit_quiz"):
            scores, sims, feedbacks = [], [], []
            with st.spinner("Grading..."):
                for i, item in enumerate(quiz):
                    correct = item.get("a", "")
                    user_ans = st.session_state["quiz_answers"][i]
                    s, sim = grade_answer(correct, user_ans)
                    scores.append(s); sims.append(sim)
                    fb_prompt = f"Student answer: {user_ans}\nCorrect answer: {correct}\nProvide a one-sentence feedback pointing out missed points or confirm it's correct (be concise)."
                    fb = call_ollama_generate(fb_prompt, timeout=30)
                    if not fb or fb.strip().startswith("⚠️"):
                        fb = "No detailed feedback available."
                    feedbacks.append(fb.strip())
                total_score = sum(scores) / len(scores) * 100.0 if len(scores) else 0.0
                st.session_state["quiz_results"] = {"scores": scores, "sims": sims, "feedbacks": feedbacks, "total_percent": round(total_score, 2)}
                st.success(f"Grading complete — Score: {round(total_score,2)}%")

        # results
        if st.session_state.get("quiz_results"):
            res = st.session_state["quiz_results"]
            st.markdown("### Results")
            st.metric("Total Score", f"{res['total_percent']}%")
            for i, item in enumerate(quiz):
                s = res["scores"][i]; sim = res["sims"][i]; fb = res["feedbacks"][i]
                st.markdown(f"**Q{i+1}. {item['q']}**")
                st.markdown(f"- **Your answer:** {st.session_state['quiz_answers'][i]}")
                st.markdown(f"- **Expected:** {item['a']}")
                st.markdown(f"- **Score:** {round(s*100,1)}% (similarity={round(sim,3)})")
                st.markdown(f"- **Feedback:** {fb}")
                if st.button(f"Re-teach Q{i+1}", key=f"reteach_{i}"):
                    teach_prompt = f"Teach this concept simply.\nQuestion: {item['q']}\nAnswer: {item['a']}\nBe short and actionable."
                    teach_text = call_ollama_generate(teach_prompt, timeout=30)
                    st.markdown("#### Re-teaching:")
                    st.markdown(teach_text)
                    cleaned = clean_text_for_tts(teach_text)
                    audio_file = text_to_speech_sync(cleaned, voice_choice)
                    if audio_file:
                        with open(audio_file, "rb") as a:
                            st.audio(a.read(), format="audio/mp3")

            # download results as PDF
            if st.button("Download Quiz Results as PDF", key="download_quiz_pdf"):
                compiled = "Quiz Results\n\n"
                compiled += f"Score: {res['total_percent']}%\n\n"
                for i, item in enumerate(quiz):
                    compiled += f"Q{i+1}. {item['q']}\nYour answer: {st.session_state['quiz_answers'][i]}\nExpected: {item['a']}\nScore: {round(res['scores'][i]*100,1)}%\nFeedback: {res['feedbacks'][i]}\n\n"
                pdf_path = export_to_pdf("Quiz Results", compiled)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button("📄 Download Results PDF", pdf_bytes, file_name="quiz_results.pdf", mime="application/pdf", key=f"download_quiz_results_pdf_{st.session_state['last_action_id']}")

# ---------------------------
# Teaching Mode: Chat Display + Sources + Follow-ups
# ---------------------------
if mode == "Teaching Mode":
    st.markdown("## 💬 Teaching Mode — Chat with the tutor")
    # reverse order: newest first
    for idx, chat in enumerate(reversed(st.session_state["chat_history"])):
        # compute stable display_index
        display_index = len(st.session_state["chat_history"]) - 1 - idx
        st.markdown(f"<div class='chat-bubble user-bubble'><b>You:</b> {chat['user']}</div>", unsafe_allow_html=True)
        with st.expander("Assistant's response (expand)"):
            st.markdown(f"<div class='chat-bubble assistant-bubble'>{chat['assistant']}</div>", unsafe_allow_html=True)

            # show follow-ups (context aware)
            followups = chat.get("followups", []) or []
            if followups:
                st.markdown("**Suggested follow-up questions:**")
                cols_fu = st.columns(min(len(followups), 4))
                for i, fu in enumerate(followups):
                    # clicking a follow-up populates text_input via session state - use unique key
                    btn_key = f"fu_{display_index}_{i}"
                    if cols_fu[i % len(cols_fu)].button(fu, key=btn_key):
                        st.session_state["voice_query"] = fu
                        # scroll-ish: create a small success indicator
                        st.success("Follow-up loaded into input (send with Ask).")

            # show sources & highlights
            citations = chat.get("citations", []) or []
            if citations:
                st.markdown("**Sources & Highlights:**")
                for cit in citations:
                    score_pct = int(cit.get("score", 0.0) * 100)
                    st.markdown(f"- **{cit['label']}** — {cit['source']} — _score: {score_pct}%_")
                    st.markdown(f"  > {cit['snippet']}")
                    if cit.get("url") and not cit["url"].startswith("UserUpload"):
                        st.markdown(f"  [Open source]({cit['url']})")
                    else:
                        st.markdown(f"  📄 {cit['source']} (uploaded document)")


                st.markdown("---")

            # PDF download: unique per chat item
            pdf_path = export_to_pdf(chat['user'], chat['assistant'])
            try:
                with open(pdf_path, "rb") as f_pdf:
                    pdf_bytes = f_pdf.read()
                st.download_button("Download Answer as PDF", pdf_bytes, file_name=f"answer_{display_index}.pdf", mime="application/pdf", key=f"pdf_chat_{display_index}")
            except Exception as e:
                st.warning(f"Could not create PDF: {e}")

            # TTS and audio download (unique key)
            try:
                cleaned = clean_text_for_tts(chat['assistant'])
                audio_file = text_to_speech_sync(cleaned, voice_choice)
                if audio_file:
                    with open(audio_file, "rb") as a:
                        audio_bytes = a.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button("Download Answer Audio", audio_bytes, file_name=f"answer_{display_index}.mp3", mime="audio/mp3", key=f"audio_chat_{display_index}")
            except Exception as e:
                st.warning(f"TTS failed for this answer: {e}")

    st.markdown("---")
    st.info("Tip: Use follow-up suggestions to continue the conversation. Enable live web snippets to include external evidence in answers.")

# ============================================================
# STEP 13 — RAG + AGENT HYBRID MODE (FINAL)
# ============================================================

if "agent_mode" not in st.session_state:
    st.session_state["agent_mode"] = False
if "agent_trace" not in st.session_state:
    st.session_state["agent_trace"] = []

st.markdown("---")
st.header("🧠 Hybrid RAG + Agent Mode (Final Mode)")
st.caption("Combines retrieval, reasoning, and action for smarter, self-verifying answers.")

# --- Agent toggle ---
agent_enabled = st.checkbox("Enable Hybrid Agent Mode", value=st.session_state["agent_mode"])
st.session_state["agent_mode"] = agent_enabled

def agent_reasoning_cycle(query: str, personality: str, enable_live_search: bool, max_hops: int = 3):
    """
    Multi-hop reasoning agent that:
    1. Retrieves RAG context
    2. Drafts reasoning
    3. Detects uncertainty
    4. Performs optional live search
    5. Synthesizes and reflects final answer
    """
    trace = []
    start_time = time.time()

    # === Step 1: Retrieve RAG context ===
    trace.append("🔍 Retrieving top knowledge base chunks...")
    top_df, top_scores = find_top_k_chunks(query, TOP_RESULTS)
    context_text = "\n\n".join(top_df["text"].tolist())[:MAX_CONTEXT_CHARS]
    trace.append(f"✅ Retrieved {len(top_df)} relevant KB chunks.")

    # === Step 2: Generate initial reasoning draft ===
    reasoning_prompt = f"""
You are an autonomous AI agent reasoning through a question.
If the provided knowledge is insufficient, output 'UNCERTAIN' and what you need to find next.

Query: {query}

Knowledge Base Context:
{context_text}
"""
    draft = call_ollama_generate(reasoning_prompt)
    trace.append("🧩 Draft reasoning generated.")

    # === Step 3: Detect uncertainty ===
    if "UNCERTAIN" in draft.upper() and enable_live_search:
        trace.append("⚠️ Uncertainty detected — triggering live web search...")
        live_results = live_search_duckduckgo(query, max_results=3)
        live_context = "\n\n".join([r["snippet"] for r in live_results])
        trace.append(f"🌐 Retrieved {len(live_results)} web results for additional context.")

        hybrid_prompt = f"""
You are an expert reasoning agent combining RAG and Web search results.

User Query:
{query}

Knowledge Base:
{context_text}

Web Snippets:
{live_context}

Task:
- Combine both KB and Web snippets.
- Generate a final verified answer with inline citations (KB# / WEB#).
- End with a short summary.
"""
        final_answer = call_ollama_generate(hybrid_prompt)
        trace.append("✅ Hybrid synthesis completed using KB + Web sources.")
    else:
        final_answer = draft
        trace.append("✅ Confident KB-based reasoning — no web search required.")

    # === Step 4: Reflect & improve ===
    reflect_prompt = f"""
Reflect on your previous answer for clarity and correctness.
If any part could be phrased better or verified, improve it.

Final Answer Draft:
{final_answer}

Return an improved, concise version.
"""
    improved_answer = call_ollama_generate(reflect_prompt)
    trace.append("💭 Reflection and refinement done.")

    elapsed_time = time.time() - start_time
    trace.append(f"⏱️ Total reasoning time: {elapsed_time:.2f}s")

    return improved_answer, trace

# === Agent UI ===
if agent_enabled:
    st.markdown("### 🤖 Ask a Question in Agent Mode")
    agent_query = st.text_input("Enter your question (Agent Mode):", key="agent_query")

    if st.button("🚀 Run Hybrid Agent"):
        if not agent_query.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Agent reasoning in progress..."):
                final_answer, trace = agent_reasoning_cycle(
                    agent_query,
                    personality=personality,
                    enable_live_search=st.session_state["enable_live_search"]
                )
                st.session_state["agent_trace"] = trace

                # Optional: integrate metrics logging
                if "metrics" in st.session_state:
                    start_t = time.time() - len(trace)  # approximate
                    log_performance_metrics(start_t, final_answer, [], [])

                st.success("✅ Agent completed its reasoning process.")
                st.markdown("### 🧠 Agent's Final Answer")
                st.markdown(final_answer)

                # Optional voice playback
                cleaned = clean_text_for_tts(final_answer)
                audio_file = text_to_speech_sync(cleaned, voice_choice)
                if audio_file:
                    with open(audio_file, "rb") as a:
                        st.audio(a.read(), format="audio/mp3")

    # --- Show reasoning trace ---
    if st.session_state["agent_trace"]:
        with st.expander("🧩 View Agent Thought Trace (Debug Transparency)"):
            for step in st.session_state["agent_trace"]:
                st.markdown(f"- {step}")

# ---------------------------
# Footer
# ---------------------------
st.caption("Developed using Whisper + Llama3 + Edge-TTS + RAG + Follow-ups")

# End of file
