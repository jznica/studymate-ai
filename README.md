# 💚 ADRIAN AI — Your Study Companion

> *The RAG-powered study boyfriend you didn't know you needed.*

Upload any PDF — lecture notes, textbook chapters, research papers — and chat with **ADRIAN**, 
your warm, smart, slightly flirtatious AI study companion. He reads your docs, understands 
the context, and answers questions like your most brilliant friend.

---

## 🛠️ Tech Stack

| Layer | Tech |
|-------|------|
| **Frontend** | Streamlit |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector DB** | FAISS (local, in-memory) |
| **RAG Chain** | LangChain ConversationalRetrievalChain |
| **PDF Parsing** | LangChain PyPDFLoader |
| **Memory** | LangChain ConversationBufferMemory |

---

## 🚀 Local Setup

### 1. Clone / download the project

```bash
git clone https://github.com/YOUR_USERNAME/studymate-ai.git
cd studymate-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### 5. Use it
1. Paste your OpenAI API key in the sidebar (get one at [platform.openai.com](https://platform.openai.com))
2. Upload a PDF (lecture notes, textbook chapter, paper…)
3. Start chatting with Adrian 💚

---

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "first commit: ADRIAN AI study companion"
git remote add origin https://github.com/YOUR_USERNAME/studymate-ai.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch `main` → file `app.py`
5. Click **Deploy** 🚀

That's it — FREE hosting, no credit card needed.

### Step 3 — Optional: Store API key as a secret
In Streamlit Cloud → your app → **Settings → Secrets**:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```
Then in `app.py`, use `st.secrets["OPENAI_API_KEY"]` as a fallback.

---

## 📁 Project Structure

```
studymate-ai/
├── app.py              ← Main Streamlit app (all-in-one)
├── requirements.txt    ← Python dependencies
├── README.md           ← You are here
└── .gitignore          ← Keeps secrets safe
```

---

## 🧠 How It Works (For Your Portfolio)

```
PDF Upload
    ↓
PyPDFLoader → raw text pages
    ↓
RecursiveCharacterTextSplitter → 800-token chunks w/ 120 overlap
    ↓
OpenAI Embeddings (text-embedding-3-small) → vectors
    ↓
FAISS Vector Store → similarity search index
    ↓
User Question → embed query → retrieve top-4 chunks
    ↓
GPT-4o-mini + custom Adrian persona prompt → warm, cited answer
    ↓
ConversationBufferMemory → maintains chat history context
```

---

## 💡 Why This Impresses Recruiters

- ✅ **Real RAG pipeline** — not just a wrapper, full retrieval architecture
- ✅ **LangChain** — industry-standard LLM framework
- ✅ **FAISS** — production vector search library (used at Meta scale)
- ✅ **Streaming context** — ConversationalRetrievalChain with memory
- ✅ **Custom prompting** — personality-driven system prompt engineering
- ✅ **Deployed** — live link you can share with anyone
- ✅ **Clean UI** — custom CSS, dark theme, professional look

---

## 📬 Contact

Built by **Jenica A** — Saveetha School of Engineering, B.Tech AIML

---

*"Adrian is fictional. But the LangChain skills are very real."* 💚
