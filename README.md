🎓 Adaptive RAG-Based AI Teaching Assistant
A Multimodal Learning and Evaluation System using Whisper, Llama3, Edge-TTS, and Streamlit

📘 Overview

This project — “Adaptive RAG-based AI Teaching Assistant” — is a Data Science Course Project that builds an intelligent, multimodal tutoring system powered by Retrieval-Augmented Generation (RAG) and local AI models.

It acts as a voice-enabled personal teacher, capable of:

Understanding spoken or typed queries
Retrieving relevant knowledge from uploaded documents (PDFs, DOCX, TXT)
Generating context-grounded answers using a local Llama3 model via Ollama
Speaking the answers aloud using Edge-TTS
Generating quizzes and grading user responses automatically
Providing live web citations, performance analytics, and agent-based reasoning

🧠 Core Features

✅ Retrieval-Augmented Generation (RAG):
Fetches relevant context from the uploaded knowledge base before answering queries.
✅ Voice Interaction:
Accepts voice input (via Whisper) and speaks responses (via Edge-TTS).
✅ Document Upload & Knowledge Expansion:
Users can upload their own learning materials (PDF, DOCX, TXT) — the assistant learns from them.
✅ Smart Evaluation Mode:
Auto-generates short-answer quizzes and grades using semantic similarity (cosine similarity).
✅ Live Web Citations:
Pulls external information snippets using the DuckDuckGo Instant Answer API for factual grounding.
✅ Performance Analytics Dashboard:
Tracks user activity, quiz scores, and response quality.
✅ RAG + Agent Hybrid Mode (Final Step):
Chooses the best strategy between retrieval-based and reasoning-based responses dynamically.

🏗️ System Architecture
+-----------------------------+
|        User Input           |
| (Text / Voice via Whisper)  |
+-------------+---------------+
              |
              v
+-----------------------------+
|     RAG Engine (Ollama)     |
|  bge-m3 Embeddings + Llama3 |
+-------------+---------------+
              |
              v
+-----------------------------+
| Contextual Response Generator|
| (Citations + Live Sources)   |
+-------------+---------------+
              |
              v
+-----------------------------+
| Voice Output (Edge-TTS)     |
+-------------+---------------+
              |
              v
+-----------------------------+
| Evaluation & Analytics Mode |
+-----------------------------+

⚙️ Tech Stack
Component	- Technology Used
Frontend UI	- Streamlit
Language	- Python 3.10+
Local LLM Engine	- Ollama (Llama3 model)
Embeddings	- bge-m3
Speech Recognition	- OpenAI Whisper
Text-to-Speech	- Microsoft Edge-TTS
Vector Search	- cosine_similarity (scikit-learn)
Storage	joblib + local embeddings file
External API	- DuckDuckGo Instant Answer API
Visualization	- Streamlit charts
Total LOC (Code Lines)	~1,000 lines (single app.py)

🧩 Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/rag-ai-teaching-assistant.git
cd rag-ai-teaching-assistant

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate   # (Windows)
# or
source venv/bin/activate  # (Mac/Linux)

3️⃣ Install Required Dependencies
pip install -r requirements.txt


📄 Sample requirements.txt:

streamlit
requests
joblib
numpy
pandas
scikit-learn
PyPDF2
python-docx
sounddevice
scipy
whisper
edge-tts
asyncio
fpdf

⚙️ Setting Up Ollama
1. Install Ollama

Visit https://ollama.com/download
 and install Ollama for your OS.

2. Pull Required Models
ollama pull llama3:latest
ollama pull bge-m3:latest

3. Run Ollama Server
ollama serve


Keep this running in the background — it exposes the API on http://localhost:11434/.
🎙️ Running the App
Run the Streamlit app with:
streamlit run app.py
Then open your browser at http://localhost:8501
.

🧱 Project Structure
rag-based-ai/
│
├── app.py                        # Main Streamlit app (~1K LOC)
├── embeddings.joblib              # Stored embeddings database
├── raw_ollama_response.json       # Logs of raw model responses
├── requirements.txt
├── /uploads                       # (optional) user PDFs / DOCX
└── /data                          # (optional) sample data files


🧠 How It Works (Pipeline Summary)

Step 1: User Interaction
User either types or records a question.
If voice input → processed by Whisper → converted to text.

Step 2: Context Retrieval
The question is embedded via bge-m3.
Cosine similarity identifies the top 6 most relevant knowledge chunks.

Step 3: Generation
Context + memory + user query → sent to Llama3 (via Ollama).
Response includes inline citations like (KB#1), (WEB#2).

Step 4: Voice Response
The answer text is converted to speech via Edge-TTS.

Step 5: Evaluation Mode
Generates quizzes from your uploaded material.
Grades your answers by comparing embedding similarity.
Provides per-question feedback and overall percentage.

Step 6: Performance Analytics
Tracks metrics like total questions, average score, and chat count.

Step 7: Agent Hybrid Mode
If no relevant context found → performs web search / reasoning fallback.

📊 Sample Output

Teaching Mode:

“Explain Object-Oriented Programming.”
→ AI replies with structured explanation and references (KB#2, WEB#1).

Evaluation Mode:

Generates 5 short-answer questions from your uploaded material.

Grades them and gives feedback.

Live Citations:

Inline citations like (WEB#1) open external sources in browser.

Analytics Mode:

Dashboard showing total quizzes, average score, chat count, etc.

📈 Performance Dashboard

Metrics captured include:
Total interactions
Average response time
Number of correct quiz answers
Knowledge base size
Recent accuracy trend (line graph)

🧪 Example Run
> 🎙️ Recording... Speak now!
> ✅ Recording complete!
> You said: Explain inheritance in Java.

→ Retrieving top 6 relevant chunks (KB#2, KB#5, KB#9)
→ Generating RAG response...
→ 💬 "Inheritance in Java allows one class to acquire properties of another (KB#5)..."
→ 🗣️ Voice output generated successfully.

🧩 Key Innovations

Multimodal Interface: Text + Voice in both directions.
Live Knowledge Expansion: Dynamic learning via uploads.
Adaptive Personalities: Professor / Mentor / Tech Bro mode.
Self-Evaluation Capability: Quiz + feedback.
Citation Transparency: Inline KB/Web references.
Analytics & Insights: For educators and learners.
Agent Hybrid Layer: Balances reasoning and retrieval dynamically.

🧾 Sample Use Cases

College students learning programming subjects (e.g., OOPS, DBMS, ML).
Self-paced learners revising with personalized tutoring.
Faculty AI assistants for automated Q&A sessions.
Voice-enabled interactive learning kiosks.

🚀 Future Enhancements

Integration with FAISS / Chroma / Qdrant for scalable vector search.
Using OpenAI o1-preview or Claude 3.5 Sonnet for advanced reasoning.
Adding multilingual support for Hindi, Marathi, etc.
Deploying on Streamlit Cloud / Hugging Face Spaces.
Persistent student analytics using SQLite or TinyDB.

🏁 License

This project is open-source for educational and research purposes.
Feel free to fork, modify, and enhance with attribution.