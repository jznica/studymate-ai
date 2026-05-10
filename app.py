import streamlit as st
import os
import tempfile
from pathlib import Path
from openai import OpenAI

st.set_page_config(
    page_title="Adrian — Study Companion",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="expanded",
)

import faiss
import numpy as np
from pypdf import PdfReader

# ── CSS — clean minimal aesthetic ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f0ee !important;
    color: #1a1a1a !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #ededed !important;
    border-right: 1px solid #e0e0de !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}

/* KEEP sidebar toggle visible */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
}

#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Header ── */
.adrian-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 24px 0;
    border-bottom: 1px solid #e0e0de;
    margin-bottom: 28px;
}

.adrian-logo {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: #ededed;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.adrian-title h1 {
    font-size: clamp(1.4rem, 2.5vw, 1.75rem);
    font-weight: 600;
    letter-spacing: -0.025em;
    margin: 0 0 2px 0;
    color: #1a1a1a;
    line-height: 1.15;
}

.adrian-title p {
    font-size: 0.8rem;
    color: #888;
    margin: 0;
    font-weight: 400;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #3b82f6;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

/* ── Welcome card ── */
.welcome-card {
    background: #ededed;
    border: 1px solid #e0e0de;
    border-radius: 16px;
    padding: 40px 28px;
    margin: 16px 0;
    text-align: center;
}

.welcome-badge {
    display: inline-block;
    font-size: 11.5px;
    font-weight: 500;
    color: #3b82f6;
    margin-bottom: 16px;
}

.welcome-card h2 {
    font-size: 1.5rem;
    font-weight: 500;
    color: #1a1a1a;
    margin: 0 0 12px 0;
    letter-spacing: -0.025em;
    line-height: 1.2;
}

.welcome-card p {
    color: #666;
    font-size: 0.875rem;
    line-height: 1.6;
    margin: 0;
    font-weight: 400;
}

/* ── Stat cards ── */
.stat-card {
    background: #ededed;
    border: 1px solid #e0e0de;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}

.stat-val {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1a1a1a;
    line-height: 1;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}

.stat-lbl {
    font-size: 0.65rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── Chat bubbles ── */
.msg-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 16px;
}

.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
    font-size: 14px;
    font-weight: 600;
}

.msg-avatar.adrian { background: #1a1a1a; color: #fff; }
.msg-avatar.user { background: #3b82f6; color: #fff; }

.msg-bubble {
    max-width: 75%;
    padding: 14px 18px;
    border-radius: 14px;
    font-size: 0.9rem;
    line-height: 1.65;
    font-weight: 400;
}

.msg-bubble.adrian {
    background: #ededed;
    border: 1px solid #e0e0de;
    color: #1a1a1a;
    border-top-left-radius: 4px;
}

.msg-bubble.user {
    background: #3b82f6;
    color: #fff;
    border-top-right-radius: 4px;
}

.msg-name {
    font-size: 0.7rem;
    font-weight: 500;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #888;
}

.msg-name.user { text-align: right; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: #1a1a1a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    padding: 8px 18px !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #3b82f6 !important;
}

input[type="password"], input[type="text"] {
    background: #fff !important;
    border: 1px solid #e0e0de !important;
    color: #1a1a1a !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}

.stSuccess {
    background: #f0f7ff !important;
    border: 1px solid #3b82f644 !important;
    color: #1e40af !important;
    border-radius: 8px !important;
}

.stInfo {
    background: #fafaf9 !important;
    border: 1px solid #e0e0de !important;
    color: #555 !important;
    border-radius: 8px !important;
}

.stWarning {
    background: #fff7ed !important;
    border: 1px solid #fdba7466 !important;
    border-radius: 8px !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #d0d0ce; border-radius: 3px; }

hr { border-color: #e0e0de !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [], "chunks": [], "faiss_index": None,
        "pdf_name": None, "chunk_count": 0, "total_questions": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── PDF + RAG ─────────────────────────────────────────────────────────────────
def extract_chunks(file, chunk_size=800, overlap=100):
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n\n"
    
    words = full_text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def get_embeddings(texts, client):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in response.data]


def build_index(embeddings):
    matrix = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def search(query, index, chunks, client, k=4):
    q_emb = get_embeddings([query], client)[0]
    q_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def ask_adrian(question, context_chunks, history, client):
    context = "\n\n---\n\n".join(context_chunks)
    history_text = ""
    for msg in history[-6:]:
        role = "Student" if msg["role"] == "user" else "Adrian"
        history_text += f"{role}: {msg['content']}\n"

    system = """You are Adrian — a thoughtful, professional AI study companion.
You are clear, precise, and encouraging without being overly casual.

STRICT RULES:
- ONLY answer from the provided document context
- If something isn't in the context: "That doesn't appear to be in your document."
- Use **bold** for key terms
- Never fabricate information"""

    user_msg = f"""Document context:
{context}

Previous conversation:
{history_text}

Student's question: {question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.5,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE — Everything inline (no sidebar dependency)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="adrian-header">
    <div class="adrian-logo">
        <svg width="20" height="20" viewBox="0 0 256 256" fill="none">
            <path fill="rgb(84,84,84)" d="M 160 88 L 194 34 L 216 0 L 256 0 L 256 40 L 221.5 93.5 L 200 128 L 256 128 L 256 256 L 96 256 L 96 168 L 64.246 220 L 40 256 L 0 256 L 0 216 L 34 162 L 56 128 L 0 128 L 0 0 L 160 0 Z"/>
        </svg>
    </div>
    <div class="adrian-title">
        <h1>Adrian</h1>
        <p><span class="status-dot"></span>Document intelligence · Retrieval-augmented · Conversational</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Setup section (always visible at top of main page) ───────────────────────
with st.expander("⚙️  Setup — API key & document upload", expanded=(st.session_state.faiss_index is None)):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**1. OpenAI API Key**")
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
            help="Get yours free at platform.openai.com"
        )
        if api_key:
            st.success("✓ Connected")
        else:
            st.caption("Get one free at platform.openai.com")
    
    with col2:
        st.markdown("**2. Upload Study PDF**")
        uploaded_file = st.file_uploader(
            "PDF",
            type="pdf",
            label_visibility="collapsed"
        )
        if not uploaded_file:
            st.caption("Lecture notes, textbooks, papers...")

    if uploaded_file and api_key:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Adrian is reading your document..."):
                try:
                    client = OpenAI(api_key=api_key)
                    chunks = extract_chunks(uploaded_file)
                    
                    all_embeddings = []
                    for i in range(0, len(chunks), 50):
                        batch = chunks[i:i+50]
                        embs = get_embeddings(batch, client)
                        all_embeddings.extend(embs)
                    
                    index = build_index(all_embeddings)
                    
                    st.session_state.chunks = chunks
                    st.session_state.faiss_index = index
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.total_questions = 0
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I've finished reading **{uploaded_file.name}**. {len(chunks)} sections indexed and ready.\n\nAsk me anything about your document."
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    elif uploaded_file and not api_key:
        st.warning("Add your API key first ↑")


# ── Welcome card or chat ─────────────────────────────────────────────────────
ready = st.session_state.get("faiss_index") is not None

if not ready:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-badge">— Introducing Adrian</div>
        <h2>A study companion that<br>actually reads your documents.</h2>
        <p>
            Upload your lecture notes, textbooks, or research papers above.<br>
            Adrian indexes them with vector search and answers questions<br>
            grounded in what's actually written.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Stats row
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.chunk_count}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
    with c2: 
        st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.total_questions}</div><div class="stat-lbl">Queries</div></div>', unsafe_allow_html=True)
    with c3:
        short = (st.session_state.pdf_name or "")[:18]
        st.markdown(f'<div class="stat-card"><div class="stat-val" style="font-size:0.85rem;font-weight:500">Active</div><div class="stat-lbl">{short}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Chat messages
    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        row_cls = "msg-row user" if is_user else "msg-row"
        bub_cls = "msg-bubble user" if is_user else "msg-bubble adrian"
        ava_cls = "msg-avatar user" if is_user else "msg-avatar adrian"
        name_cls = "msg-name user" if is_user else "msg-name adrian"
        avatar = "U" if is_user else "A"
        name = "You" if is_user else "Adrian"
        content = msg["content"].replace("\n", "<br>")
        st.markdown(
            f'<div class="{row_cls}"><div class="{ava_cls}">{avatar}</div>'
            f'<div><div class="{name_cls}">{name}</div>'
            f'<div class="{bub_cls}">{content}</div></div></div>',
            unsafe_allow_html=True
        )

    # Clear button
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.rerun()

    # Chat input
    user_input = st.chat_input("Ask Adrian about your document...")
    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.total_questions += 1

        with st.spinner("Adrian is thinking..."):
            try:
                client = OpenAI(api_key=api_key)
                context_chunks = search(
                    user_input,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    client
                )
                answer = ask_adrian(
                    user_input, context_chunks,
                    st.session_state.messages[:-1], client
                )
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"An error occurred: `{str(e)[:120]}`\n\nPlease verify your API key and try again."
                })
        st.rerun()
