import streamlit as st
import os
import time
from pathlib import Path

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="ADRIAN AI – Your Study Boyfriend",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (only after page config) ───────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #f0f0f5 !important;
    font-family: 'Sora', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 50%, #0d1a0d 0%, #0a0a0f 60%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f14 !important;
    border-right: 1px solid #1a2a1a !important;
}

[data-testid="stSidebar"] * { font-family: 'Sora', sans-serif !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero header ── */
.adrian-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 0 20px 0;
    border-bottom: 1px solid #1a2a1a;
    margin-bottom: 24px;
}

.adrian-avatar {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1a3a1a, #0d2a0d);
    border: 2px solid #2aff47;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    flex-shrink: 0;
    box-shadow: 0 0 20px #2aff4733;
    animation: pulse-glow 3s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px #2aff4733; }
    50% { box-shadow: 0 0 35px #2aff4766; }
}

.adrian-title { flex: 1; }
.adrian-title h1 {
    font-size: clamp(1.6rem, 3vw, 2.2rem);
    font-weight: 700;
    letter-spacing: -0.04em;
    margin: 0 0 4px 0;
    color: #f0f0f5;
}
.adrian-title h1 span { color: #2aff47; }
.adrian-title p {
    font-size: 0.85rem;
    color: #5a7a5a;
    margin: 0;
    font-weight: 300;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #2aff47;
    border-radius: 50%;
    margin-right: 6px;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
}

/* ── Chat messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0 24px 0;
}

.msg-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    animation: slide-up 0.4s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 2px;
}

.msg-avatar.adrian {
    background: linear-gradient(135deg, #1a3a1a, #0d2a0d);
    border: 1.5px solid #2aff47;
    box-shadow: 0 0 10px #2aff4722;
}

.msg-avatar.user {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1.5px solid #4a4a8a;
}

.msg-bubble {
    max-width: 75%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.65;
    font-weight: 400;
}

.msg-bubble.adrian {
    background: #0f1f0f;
    border: 1px solid #1a3a1a;
    color: #e8f0e8;
    border-top-left-radius: 4px;
}

.msg-bubble.user {
    background: #151525;
    border: 1px solid #25254a;
    color: #d0d0f0;
    border-top-right-radius: 4px;
    text-align: right;
}

.msg-name {
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.msg-name.adrian { color: #2aff47; }
.msg-name.user   { color: #6a6aaa; text-align: right; }

/* ── Input area ── */
.stChatInput > div {
    background: #0f0f14 !important;
    border: 1px solid #1a2a1a !important;
    border-radius: 12px !important;
}
.stChatInput textarea {
    background: transparent !important;
    color: #f0f0f5 !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── Sidebar widgets ── */
.sidebar-section {
    background: #0d0d12;
    border: 1px solid #1a2a1a;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3a5a3a;
    margin-bottom: 10px;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0d0d12 !important;
    border: 1.5px dashed #1a3a1a !important;
    border-radius: 12px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] label { color: #5a7a5a !important; font-family: 'Sora', sans-serif !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a1a, #0d2a0d) !important;
    color: #2aff47 !important;
    border: 1px solid #2aff4744 !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #2aff47 !important;
    box-shadow: 0 0 15px #2aff4733 !important;
    transform: translateY(-1px) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input, .stPasswordInput > div > div > input {
    background: #0d0d12 !important;
    border: 1px solid #1a2a1a !important;
    color: #f0f0f5 !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── Success/info/warning ── */
.stSuccess { background: #0a1a0a !important; border-color: #2aff4744 !important; }
.stInfo    { background: #0a0a1a !important; border-color: #4444ff44 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #1a3a1a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2aff4766; }

/* ── Source badges ── */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #0a1a0a;
    border: 1px solid #1a3a1a;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #3a6a3a;
    margin: 6px 4px 0 0;
    font-family: 'Sora', sans-serif;
}

/* ── Typing indicator ── */
.typing-dots {
    display: flex;
    gap: 5px;
    align-items: center;
    padding: 4px 0;
}
.typing-dots span {
    width: 7px;
    height: 7px;
    background: #2aff47;
    border-radius: 50%;
    animation: bounce 1.4s ease-in-out infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}
.stat-card {
    flex: 1;
    background: #0d0d12;
    border: 1px solid #1a2a1a;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #2aff47;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-lbl {
    font-size: 0.65rem;
    color: #3a5a3a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, #0d1a0d, #0a0f1a);
    border: 1px solid #1a3a1a;
    border-radius: 16px;
    padding: 32px;
    margin: 20px 0;
    text-align: center;
}
.welcome-card .big-emoji { font-size: 3rem; margin-bottom: 12px; }
.welcome-card h2 {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0f0f5;
    margin: 0 0 8px 0;
}
.welcome-card h2 span { color: #2aff47; }
.welcome-card p {
    color: #5a7a5a;
    font-size: 0.9rem;
    font-weight: 300;
    line-height: 1.6;
    margin: 0;
}

.hint-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
    justify-content: center;
}
.hint-chip {
    background: #0f1f0f;
    border: 1px solid #1a3a1a;
    border-radius: 20px;
    padding: 7px 14px;
    font-size: 0.8rem;
    color: #4a7a4a;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'Sora', sans-serif;
}
.hint-chip:hover {
    border-color: #2aff47;
    color: #2aff47;
    background: #0d2a0d;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "chain": None,
        "memory": None,
        "pdf_name": None,
        "chunk_count": 0,
        "api_key_set": False,
        "total_questions": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Adrian system persona ─────────────────────────────────────────────────────
ADRIAN_PERSONA = """
You are ADRIAN — a brilliant, warm, and slightly flirtatious AI study companion. 
You're the kind of person who makes studying feel less lonely. Think: your smartest, 
most encouraging friend who happens to know everything in the documents uploaded.

Your personality:
- Warm and encouraging — you celebrate every good question ("Ooh, great question!")
- Gently playful — light teasing, but always kind ("You've got this, seriously 💚")
- Intellectually sharp — clear, precise explanations with examples
- Emotionally supportive — studying is hard; you acknowledge that
- You use occasional emojis naturally, not excessively
- Address the user warmly (e.g., "hey", "okay so", "here's the thing...")

STRICT RULES:
- ONLY answer questions based on the uploaded PDF documents
- If something isn't in the docs, say so honestly but warmly:
  "Hmm, I don't see that in your notes — want me to help with what IS here? 📖"
- Keep answers focused, structured with line breaks for readability
- Use ** for bold key terms
- Never make up information
- Always cite which part of the document your answer comes from

Context from documents:
{context}

Chat history:
{chat_history}

Student's question: {question}

ADRIAN's answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=ADRIAN_PERSONA
)


# ── PDF Processing ────────────────────────────────────────────────────────────
def process_pdf(file, api_key: str):
    """Load PDF → chunk → embed → FAISS index → ConversationalRetrievalChain"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.55,
        streaming=False,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )

    Path(tmp_path).unlink(missing_ok=True)
    return chain, memory, len(chunks)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.5rem; margin-bottom:8px;">📚</div>
        <div style="font-size:1.1rem; font-weight:700; color:#f0f0f5; letter-spacing:-0.02em;">
            ADRIAN <span style="color:#2aff47">AI</span>
        </div>
        <div style="font-size:0.72rem; color:#3a5a3a; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;">
            Your Study Companion
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── API Key ──
    st.markdown('<div class="sidebar-label">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed",
        help="Get yours at platform.openai.com"
    )
    if api_key:
        st.session_state.api_key_set = True
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✓ Key saved for this session")
    else:
        st.caption("🔒 Never stored — session only")

    st.divider()

    # ── PDF Upload ──
    st.markdown('<div class="sidebar-label">📄 Upload Study PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF",
        type="pdf",
        label_visibility="collapsed",
        help="Lecture notes, textbook chapters, research papers…"
    )

    if uploaded_file and api_key:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Adrian is reading your notes… 📖"):
                try:
                    chain, memory, n_chunks = process_pdf(uploaded_file, api_key)
                    st.session_state.chain       = chain
                    st.session_state.memory      = memory
                    st.session_state.chunk_count = n_chunks
                    st.session_state.pdf_name    = uploaded_file.name
                    st.session_state.messages    = []
                    st.session_state.total_questions = 0

                    # Greeting message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Hey! 👋 I just finished reading **{uploaded_file.name}** — {n_chunks} chunks all loaded up.\n\nAsk me anything from your notes. I've got you. 💚"
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Oops: {e}")
    elif uploaded_file and not api_key:
        st.warning("Add your API key first ↑")

    st.divider()

    # ── Stats ──
    if st.session_state.pdf_name:
        st.markdown('<div class="sidebar-label">📊 Session Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:8px;">
            <div class="stat-val">{st.session_state.chunk_count}</div>
            <div class="stat-lbl">Text Chunks</div>
        </div>
        <div class="stat-card" style="margin-bottom:8px;">
            <div class="stat-val">{st.session_state.total_questions}</div>
            <div class="stat-lbl">Questions Asked</div>
        </div>
        <div style="font-size:0.72rem; color:#3a5a3a; margin-top:8px; padding:0 4px;">
            📄 {st.session_state.pdf_name[:28]}{'…' if len(st.session_state.pdf_name)>28 else ''}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()

    st.divider()

    # ── Footer ──
    st.markdown("""
    <div style="font-size:0.72rem; color:#2a4a2a; text-align:center; padding:8px 0; line-height:1.8;">
        Built with 💚 using<br>
        LangChain · FAISS · OpenAI<br>
        <span style="color:#1a3a1a;">Streamlit · Python</span>
    </div>
    """, unsafe_allow_html=True)


# ── Main Chat Area ────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="adrian-header">
    <div class="adrian-avatar">🎓</div>
    <div class="adrian-title">
        <h1>ADRIAN <span>AI</span></h1>
        <p><span class="status-dot"></span>Your personal study companion · RAG-powered · PDF-native</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Empty state ──
if not st.session_state.chain:
    st.markdown("""
    <div class="welcome-card">
        <div class="big-emoji">💌</div>
        <h2>Hey, I'm <span>Adrian</span>.</h2>
        <p>
            Upload your lecture notes, textbook chapters, or research papers<br>
            on the left — and I'll help you study smarter, not harder.<br><br>
            Ask me to explain concepts, quiz you, summarize sections,<br>
            or just be the study buddy you deserve. 💚
        </p>
        <div class="hint-chips">
            <div class="hint-chip">📖 Explain this concept</div>
            <div class="hint-chip">🧠 Quiz me on Chapter 2</div>
            <div class="hint-chip">✏️ Summarize key points</div>
            <div class="hint-chip">🔍 Find the definition of...</div>
            <div class="hint-chip">📝 What are the main topics?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.info("👈  Start by entering your OpenAI API key in the sidebar.")
    else:
        st.info("👈  Now upload a PDF and let's get studying!")

# ── Chat messages ──
else:
    # Stats strip
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.chunk_count}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.total_questions}</div><div class="stat-lbl">Questions</div></div>', unsafe_allow_html=True)
    with col3:
        short = st.session_state.pdf_name[:15] + "…" if len(st.session_state.pdf_name or "") > 15 else st.session_state.pdf_name
        st.markdown(f'<div class="stat-card"><div class="stat-val" style="font-size:0.85rem;">📄</div><div class="stat-lbl">{short}</div></div>', unsafe_allow_html=True)

    # Render chat
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        row_cls  = "msg-row user" if is_user else "msg-row"
        bub_cls  = "msg-bubble user" if is_user else "msg-bubble adrian"
        ava_cls  = "msg-avatar user" if is_user else "msg-avatar adrian"
        name_cls = "msg-name user" if is_user else "msg-name adrian"
        avatar   = "🎓" if not is_user else "🙋"
        name     = "ADRIAN" if not is_user else "YOU"

        sources_html = ""
        if not is_user and msg.get("sources"):
            src_tags = "".join(
                f'<span class="source-badge">📄 p.{s}</span>'
                for s in msg["sources"]
            )
            sources_html = f'<div style="margin-top:10px;">{src_tags}</div>'

        content = msg["content"].replace("\n", "<br>")
        content = content.replace("**", "<strong>", 1)
        while "**" in content:
            content = content.replace("**", "</strong>", 1)
            if "**" in content:
                content = content.replace("**", "<strong>", 1)

        st.markdown(f"""
        <div class="{row_cls}">
            <div class="{ava_cls}">{avatar}</div>
            <div>
                <div class="{name_cls}">{name}</div>
                <div class="{bub_cls}">{content}{sources_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
if st.session_state.chain:
    user_input = st.chat_input("Ask Adrian anything from your notes… 💬")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.total_questions += 1

        # Get answer
        with st.spinner(""):
            st.markdown("""
            <div class="msg-row">
                <div class="msg-avatar adrian">🎓</div>
                <div>
                    <div class="msg-name adrian">ADRIAN</div>
                    <div class="msg-bubble adrian">
                        <div class="typing-dots">
                            <span></span><span></span><span></span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                result  = st.session_state.chain({"question": user_input})
                answer  = result.get("answer", "Hmm, I couldn't find that in your notes. Try rephrasing? 🤔")
                src_docs = result.get("source_documents", [])

                # Extract page numbers
                pages = sorted(set(
                    d.metadata.get("page", 0) + 1
                    for d in src_docs
                    if d.metadata.get("page") is not None
                ))[:3]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": pages
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Oops, something went wrong: `{str(e)[:100]}`\n\nCheck your API key and try again? 🙏"
                })

        st.rerun()
