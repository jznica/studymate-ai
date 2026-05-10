import streamlit as st
import os
from openai import OpenAI

st.set_page_config(
    page_title="Adrian — Study Companion",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="expanded",
)

import numpy as np
from pypdf import PdfReader
import re
from collections import Counter

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#f0f0ee !important;color:#1a1a1a !important;font-family:'Inter',sans-serif !important}
[data-testid="stSidebar"]{background:#ededed !important;border-right:1px solid #e0e0de !important}
#MainMenu,footer{visibility:hidden}
.stDeployButton{display:none}
.adrian-header{display:flex;align-items:center;gap:16px;padding:28px 0 24px 0;border-bottom:1px solid #e0e0de;margin-bottom:28px}
.adrian-logo{width:44px;height:44px;border-radius:50%;background:#ededed;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.adrian-title h1{font-size:1.6rem;font-weight:600;letter-spacing:-0.025em;margin:0 0 2px 0;color:#1a1a1a;line-height:1.15}
.adrian-title p{font-size:0.8rem;color:#888;margin:0;font-weight:400}
.status-dot{width:6px;height:6px;background:#3b82f6;border-radius:50%;display:inline-block;margin-right:6px}
.welcome-card{background:#ededed;border:1px solid #e0e0de;border-radius:16px;padding:40px 28px;margin:16px 0;text-align:center}
.welcome-badge{display:inline-block;font-size:11.5px;font-weight:500;color:#3b82f6;margin-bottom:16px}
.welcome-card h2{font-size:1.5rem;font-weight:500;color:#1a1a1a;margin:0 0 12px 0;letter-spacing:-0.025em;line-height:1.2}
.welcome-card p{color:#666;font-size:0.875rem;line-height:1.6;margin:0;font-weight:400}
.stat-card{background:#ededed;border:1px solid #e0e0de;border-radius:12px;padding:14px;text-align:center}
.stat-val{font-size:1.4rem;font-weight:600;color:#1a1a1a;line-height:1;margin-bottom:4px;letter-spacing:-0.02em}
.stat-lbl{font-size:0.65rem;color:#888;text-transform:uppercase;letter-spacing:0.08em;font-weight:500}
.msg-row{display:flex;gap:12px;align-items:flex-start;margin-bottom:16px}
.msg-row.user{flex-direction:row-reverse}
.msg-avatar{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;font-size:14px;font-weight:600}
.msg-avatar.adrian{background:#1a1a1a;color:#fff}
.msg-avatar.user{background:#3b82f6;color:#fff}
.msg-bubble{max-width:75%;padding:14px 18px;border-radius:14px;font-size:0.9rem;line-height:1.65;font-weight:400}
.msg-bubble.adrian{background:#ededed;border:1px solid #e0e0de;color:#1a1a1a;border-top-left-radius:4px}
.msg-bubble.user{background:#3b82f6;color:#fff;border-top-right-radius:4px}
.msg-name{font-size:0.7rem;font-weight:500;margin-bottom:6px;letter-spacing:0.05em;text-transform:uppercase;color:#888}
.msg-name.user{text-align:right}
.stButton>button{background:#1a1a1a !important;color:#fff !important;border:none !important;border-radius:999px !important;font-family:'Inter',sans-serif !important;font-weight:500 !important;font-size:0.8rem !important;padding:8px 18px !important;width:100% !important}
.stButton>button:hover{background:#3b82f6 !important}
input[type="password"],input[type="text"]{background:#fff !important;border:1px solid #e0e0de !important;color:#1a1a1a !important;border-radius:8px !important}
.stSuccess{background:#f0f7ff !important;border:1px solid #3b82f644 !important;color:#1e40af !important;border-radius:8px !important}
.stInfo{background:#fafaf9 !important;border:1px solid #e0e0de !important;color:#555 !important;border-radius:8px !important}
.stWarning{background:#fff7ed !important;border:1px solid #fdba7466 !important;border-radius:8px !important}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-thumb{background:#d0d0ce;border-radius:3px}
hr{border-color:#e0e0de !important}
</style>
""", unsafe_allow_html=True)


# ── OpenRouter config ────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Fallback list — Adrian tries these in order until one isn't rate-limited
FREE_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-coder:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]


def get_openrouter_client(api_key):
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# ── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [], "chunks": [], "chunk_keywords": [],
        "pdf_name": None, "chunk_count": 0, "total_questions": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── PDF chunking ─────────────────────────────────────────────────────────────
def extract_chunks(file, chunk_size=600, overlap=80):
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        try:
            full_text += page.extract_text() + "\n\n"
        except Exception:
            continue
    words = full_text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# ── Lightweight retrieval ────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


def build_chunk_index(chunks):
    return [Counter(tokenize(c)) for c in chunks]


def search_chunks(query, chunks, chunk_keywords, k=4):
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return chunks[:k]
    scores = []
    for i, kw_dict in enumerate(chunk_keywords):
        score = sum(kw_dict.get(t, 0) for t in q_tokens)
        scores.append((score, i))
    scores.sort(reverse=True)
    top_idx = [i for _, i in scores[:k]]
    return [chunks[i] for i in top_idx]


# ── Chat function with auto-fallback across models ──────────────────────────
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

    last_error = None
    for model in FREE_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.5,
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue

    raise Exception(f"All free models are rate-limited right now. Try again in a minute. ({str(last_error)[:100]})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
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
        <p><span class="status-dot"></span>Document intelligence · Free LLM via OpenRouter · Conversational</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Setup section ─────────────────────────────────────────────────────────────
with st.expander("⚙️  Setup — API key & document upload", expanded=(not st.session_state.chunks)):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**1. OpenRouter API Key**")
        api_key = st.text_input(
            "API Key", type="password", placeholder="sk-or-v1-...",
            label_visibility="collapsed",
            help="Free at openrouter.ai — no credit card needed"
        )
        if api_key:
            if api_key.startswith("sk-or-"):
                st.success("✓ Connected")
            else:
                st.warning("⚠ Should start with sk-or-")
        else:
            st.caption("Get free key at openrouter.ai/keys")

    with col2:
        st.markdown("**2. Upload Study PDF**")
        uploaded_file = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
        if not uploaded_file:
            st.caption("Lecture notes, textbooks, papers...")

    if uploaded_file and api_key:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Adrian is reading your document..."):
                try:
                    chunks = extract_chunks(uploaded_file)
                    chunk_keywords = build_chunk_index(chunks)
                    st.session_state.chunks = chunks
                    st.session_state.chunk_keywords = chunk_keywords
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
                    st.error(f"Error reading PDF: {e}")
    elif uploaded_file and not api_key:
        st.warning("Add your API key first ↑")


# ── Welcome / chat ──────────────────────────────────────────────────────────
ready = bool(st.session_state.chunks)

if not ready:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-badge">— Introducing Adrian</div>
        <h2>A study companion that<br>actually reads your documents.</h2>
        <p>
            Upload your lecture notes, textbooks, or research papers above.<br>
            Adrian indexes them and answers questions grounded<br>
            in what's actually written. Powered by free LLMs.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.chunk_count}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.total_questions}</div><div class="stat-lbl">Queries</div></div>', unsafe_allow_html=True)
    with c3:
        short = (st.session_state.pdf_name or "")[:18]
        st.markdown(f'<div class="stat-card"><div class="stat-val" style="font-size:0.85rem;font-weight:500">Active</div><div class="stat-lbl">{short}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        row_cls = "msg-row user" if is_user else "msg-row"
        bub_cls = "msg-bubble user" if is_user else "msg-bubble adrian"
        ava_cls = "msg-avatar user" if is_user else "msg-avatar adrian"
        name_cls = "msg-name user" if is_user else "msg-name adrian"
        avatar = "U" if is_user else "A"
        name = "You" if is_user else "Adrian"
        import re as _re
        content = msg["content"]
        content = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = content.replace("\n", "<br>")
        st.markdown(
            f'<div class="{row_cls}"><div class="{ava_cls}">{avatar}</div>'
            f'<div><div class="{name_cls}">{name}</div>'
            f'<div class="{bub_cls}">{content}</div></div></div>',
            unsafe_allow_html=True
        )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.rerun()

    user_input = st.chat_input("Ask Adrian about your document...")
    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.total_questions += 1

        with st.spinner("Adrian is thinking..."):
            try:
                client = get_openrouter_client(api_key)
                context_chunks = search_chunks(
                    user_input, st.session_state.chunks, st.session_state.chunk_keywords
                )
                answer = ask_adrian(
                    user_input, context_chunks,
                    st.session_state.messages[:-1], client
                )
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: `{str(e)[:200]}`"
                })
        st.rerun()
