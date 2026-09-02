# 🧠 Obsidian Vault Knowledge Assistant

A privacy-focused, local-first Retrieval-Augmented Generation (RAG) web application that empowers users to query their Obsidian Markdown notes in natural language, receiving factual, grounded answers complete with inline source note citations and interactive snippet previews.

[![Live Demo](https://img.shields.io/badge/Streamlit_Cloud-Live_Demo-FF4B4B?logo=streamlit)](https://share.streamlit.io/shagunvishnoi/obsidian-rag-assistant/main/app.py)

---

## 🏗️ Architecture Diagram

```
 ┌────────────────────────┐
 │   Obsidian Vault       │
 │   (.md files upload    │
 │  or Sample Vault)      │
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │  Markdown Parser &     │  ---> Strips [[wikilinks]], extracts headers
 │  Recursive Chunker     │  ---> Splits into ~300-500 token overlapping chunks
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │  Sentence-Transformers │  ---> all-MiniLM-L6-v2 (384-d dense vectors)
 │  (Local Embeddings)    │
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │       ChromaDB         │  ---> Local persistent vector storage
 │   (Vector Database)    │
 └───────────┬────────────┘
             │ (Cosine Similarity Search, top-k=4)
             ▼
 ┌────────────────────────┐
 │   Prompt Builder &     │  ---> Context grounding & citation enforcement
 │   Groq LLM Service     │  ---> Llama-3.1-8b-instant
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │  Streamlit Chat UI &   │  ---> Interactive chat interface with expandable
 │  Source Citations      │       source excerpts & similarity scores
 └────────────────────────┘
```

---

## 🛠️ Tech Stack & Design Rationale

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend & Backend** | `Streamlit` | Enables rapid, Python-native development of responsive chat UIs without separate node/web server infrastructure. |
| **LLM Provider** | `Groq SDK` (`llama-3.1-8b-instant`) | Delivers sub-second inference speeds on a generous free-tier API rate limit. |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Generates high-quality 384-dimensional semantic embeddings locally without external API costs or rate limits. |
| **Vector Store** | `ChromaDB` (`PersistentClient`) | Lightweight, open-source, serverless local vector database that persists document vectors and metadata cleanly on disk. |
| **Environment Config** | `python-dotenv` & `st.secrets` | Ensures seamless local development via `.env` while maintaining full compatibility with Streamlit Community Cloud secret deployment. |

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10 – 3.13 installed
- A free Groq API key from [Groq Console](https://console.groq.com)

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/shagunvishnoi/obsidian-rag-assistant.git
cd obsidian-rag-assistant

python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💡 RAG Approach & Implementation Rationale

### 1. Why Retrieval-Augmented Generation (RAG)?
Personal Obsidian vaults grow over time into complex networks of ideas. Direct LLM prompts are limited by context windows and lack private note awareness. RAG provides semantic precision, dynamic knowledge updates without costly retraining, and zero-hallucination factual grounding.

### 2. Chunking Strategy
Personal notes vary from short bullet lists to lengthy long-form essays. We employ a **Recursive Character Chunking** strategy:
- **Target Size**: ~1,400 characters (~350 tokens), ideal for sentence transformer context windows.
- **Overlap**: ~200 characters (~50 tokens) to guarantee semantic context continuity across chunk boundaries.
- **Header Section Context**: Each chunk tracks its nearest parent Markdown header (`#`, `##`, `###`), attaching header tags directly to metadata.

### 3. Embedding Model Selection
`all-MiniLM-L6-v2` was selected because it balances fast CPU execution with strong benchmark retrieval performance. Its 384-dimensional vector space yields crisp cosine similarity scores over note excerpts.

### 4. Citation Verification & Anti-Hallucination Guardrails
To enforce factual accuracy:
1. System prompts mandate that answers must be derived **strictly** from retrieved context snippets.
2. If context similarity falls short or fails to contain the answer, the model explicitly responds with: *"I don't have enough information in your notes to answer this."*
3. Each retrieved chunk displays source note filenames, section headings, similarity scores, and expandable full-text previews directly beneath the generated answer.

---

## 🤖 AI Usage Disclosure

This project was built using AI pair-programming tools (Google Antigravity / Gemini) for initial boilerplates and test scaffolding. All underlying system architecture, chunking pipelines, ChromaDB schema design, Groq integration rules, prompt engineering guardrails, and error handling mechanisms were designed, reviewed, and verified by the author.

---

## 🌐 Deploying to Streamlit Community Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit for Obsidian RAG Knowledge Assistant"
   git push -u origin main
   ```
2. **Connect Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Click **"New App"** and select repository `shagunvishnoi/obsidian-rag-assistant`.
   - Set Main file path to `app.py`.
3. **Configure Secrets**:
   - In the Streamlit Cloud app settings, open the **Secrets** section.
   - Add:
     ```toml
     GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
     ```
4. **Deploy**: Click **Deploy!**

---

## 📸 Application Screenshots

*(Screenshots will be added following deployment)*
