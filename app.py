import streamlit as st
import os
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="ADRIAN AI – Your Study Boyfriend",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important; color: #f0f0f5 !important;
    font-family: 'Sora', sans-serif !important;
}
[data-testid="stSidebar"] { background: #0f0f14 !important; border-right: 1px solid #1a2a1a !important; }
#MainMenu, footer, header { visibility: hidden; }
.adrian-header { display:flex; align-items:center; gap:18px; padding:28px 0 20px 0; border-bottom:1px solid #1a2a1a; margin-bottom:24px; }
.adrian-avatar { width:64px; height:64px; border-radius:50%; background:linear-gradient(135deg,#1a3a1a,#0d2a0d); border:2px solid #2aff47; display:flex; align-items:center; justify-content:center; font-size:28px; }
.adrian-title h1 { font-size:clamp(1.6rem,3vw,2.2rem); font-weight:700; letter-spacing:-0.04em; margin:0 0 4px 0; color:#f0f0f5; }
.adrian-title h1 span { color:#2aff47; }
.adrian-title p { font-size:0.85rem; color:#5a7a5a; margin:0; }
.status-dot { display:inline-block; width:8px; height:8px; background:#2aff47; border-radius:50%; margin-right:6px; }
.msg-row { display:flex; gap:12px; align-items:flex-start; margin-bottom:16px; }
.msg-row.user { flex-direction:row-reverse; }
.msg-avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:16px; flex-shrink:0; margin-top:2px; }
.msg-avatar.adrian { background:linear-gradient(135deg,#1a3a1a,#0d2a0d); border:1.5px solid #2aff47; }
.msg-avatar.user { background:linear-gradient(135deg,#1a1a2e,#16213e); border:1.5px solid #4a4a8a; }
.msg-bubble { max-width:75%; padding:14px 18px; border-radius:16px; font-size:0.9rem; line-height:1.65; }
.msg-bubble.adrian { background:#0f1f0f; border:1px solid #1a3a1a; color:#e8f0e8; border-top-left-radius:4px; }
.msg-bubble.user { background:#151525; border:1px solid #25254a; color:#d0d0f0; border-top-right-radius:4px; }
.msg-name { font-size:0.72rem; font-weight:600; margin-bottom:6px; letter-spacing:0.05em; text-transform:uppercase; }
.msg-name.adrian { color:#2aff47; }
.msg-name.user { color:#6a6aaa; }
.source-badge { display:inline-flex; align-items:center; background:#0a1a0a; border:1px solid #1a3a1a; border-radius:6px; padding:3px 10px; font-size:0.72rem; color:#3a6a3a; margin:6px 4px 0 0; }
.stat-card { background:#0d0d12; border:1px solid #1a2a1a; border-radius:10px; padding:12px; text-align:center; }
.stat-val { font-size:1.4rem; font-weight:700; color:#2aff47; line-height:1; margin-bottom:4px; }
.stat-lbl { font-size:0.65rem; color:#3a5a3a; text-transform:uppercase; letter-spacing:0.08em; }
.welcome-card { background:linear-gradient(135deg,#0d1a0d,#0a0f1a); border:1px solid #1a3a1a; border-radius:16px; padding:32px; margin:20px 0; text-align:center; }
.welcome-card h2 { font-size:1.4rem; font-weight:600; color:#f0f0f5; margin:0 0 8px 0; }
.welcome-card h2 span { color:#2aff47; }
.welcome-card p { color:#5a7a5a; font-size:0.9rem; line-height:1.6; margin:0; }
.stButton > button { background:linear-gradient(135deg,#1a3a1a,#0d2a0d) !important; color:#2aff47 !important; border:1px solid #2aff4744 !important; border-radius:8px !important; font-family:'Sora',sans-serif !important; font-weight:600 !important; }
.stTextInput > div > div > input { background:#0d0d12 !important; border:1px solid #1a2a1a !important; color:#f0f0f5 !important; border-radius:8px !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:#1a3a1a; border-radius:2px; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {"messages":[],"chain":None,"memory":None,"pdf_name":None,"chunk_count":0,"api_key_set":False,"total_questions":0}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

ADRIAN_PERSONA = """You are ADRIAN — a brilliant, warm, and encouraging AI study companion.
You make studying feel less lonely. You are the smartest, most encouraging friend who knows everything in the uploaded documents.

Personality:
- Warm and encouraging — celebrate every good question
- Gently playful but always kind
- Intellectually sharp — clear, precise explanations
- Use occasional emojis naturally

STRICT RULES:
- ONLY answer from the uploaded PDF documents
- If not in the docs: "Hmm, I don't see that in your notes — want me to help with what IS here?"
- Use ** for bold key terms
- Never make up information

Context: {context}
Chat history: {chat_history}
Question: {question}
ADRIAN's answer:"""

PROMPT = PromptTemplate(input_variables=["context","chat_history","question"], template=ADRIAN_PERSONA)


def process_pdf(file, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=0.55)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k":4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )
    Path(tmp_path).unlink(missing_ok=True)
    return chain, memory, len(chunks)


with st.sidebar:
    st.markdown('<div style="text-align:center;padding:20px 0 10px 0;"><div style="font-size:2.5rem">📚</div><div style="font-size:1.1rem;font-weight:700;color:#f0f0f5;">ADRIAN <span style="color:#2aff47">AI</span></div><div style="font-size:0.7rem;color:#3a5a3a;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">Your Study Companion</div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p style="font-size:0.7rem;font-weight:600;text-transform:uppercase;color:#3a5a3a;">🔑 OpenAI API Key</p>', unsafe_allow_html=True)
    api_key = st.text_input("API Key", type="password", placeholder="sk-...", label_visibility="collapsed")
    if api_key:
        st.session_state.api_key_set = True
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✓ Key saved for this session")
    else:
        st.caption("🔒 Never stored — session only")
    st.divider()
    st.markdown('<p style="font-size:0.7rem;font-weight:600;text-transform:uppercase;color:#3a5a3a;">📄 Upload Study PDF</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file and api_key:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Adrian is reading your notes… 📖"):
                try:
                    chain, memory, n = process_pdf(uploaded_file, api_key)
                    st.session_state.chain = chain
                    st.session_state.memory = memory
                    st.session_state.chunk_count = n
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.total_questions = 0
                    st.session_state.messages.append({"role":"assistant","content":f"Hey! 👋 I just finished reading **{uploaded_file.name}** — {n} chunks loaded.\n\nAsk me anything from your notes. I've got you. 💚"})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    elif uploaded_file and not api_key:
        st.warning("Add your API key first ↑")
    st.divider()
    if st.session_state.pdf_name:
        st.markdown('<p style="font-size:0.7rem;font-weight:600;text-transform:uppercase;color:#3a5a3a;">📊 Stats</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.chunk_count}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.total_questions}</div><div class="stat-lbl">Questions</div></div>', unsafe_allow_html=True)
        st.caption(f"📄 {st.session_state.pdf_name[:30]}")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            if st.session_state.memory: st.session_state.memory.clear()
            st.rerun()
    st.divider()
    st.markdown('<div style="font-size:0.7rem;color:#2a4a2a;text-align:center;line-height:1.8;">Built with 💚<br>LangChain · FAISS · OpenAI<br>Streamlit · Python</div>', unsafe_allow_html=True)


st.markdown('<div class="adrian-header"><div class="adrian-avatar">🎓</div><div class="adrian-title"><h1>ADRIAN <span>AI</span></h1><p><span class="status-dot"></span>Your personal study companion · RAG-powered · PDF-native</p></div></div>', unsafe_allow_html=True)

if not st.session_state.chain:
    st.markdown('<div class="welcome-card"><div style="font-size:3rem;margin-bottom:12px;">💌</div><h2>Hey, I\'m <span>Adrian</span>.</h2><p>Upload your lecture notes, textbook chapters, or research papers on the left<br>and I\'ll help you study smarter, not harder.<br><br>Ask me to explain concepts, quiz you, or summarize sections. 💚</p></div>', unsafe_allow_html=True)
    if not api_key:
        st.info("👈  Start by entering your OpenAI API key in the sidebar.")
    else:
        st.info("👈  Now upload a PDF and let's get studying!")
else:
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.chunk_count}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.total_questions}</div><div class="stat-lbl">Questions</div></div>', unsafe_allow_html=True)
    with c3:
        short = (st.session_state.pdf_name or "")[:14]
        st.markdown(f'<div class="stat-card"><div class="stat-val" style="font-size:0.85rem">📄</div><div class="stat-lbl">{short}</div></div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        row_cls = "msg-row user" if is_user else "msg-row"
        bub_cls = "msg-bubble user" if is_user else "msg-bubble adrian"
        ava_cls = "msg-avatar user" if is_user else "msg-avatar adrian"
        name_cls = "msg-name user" if is_user else "msg-name adrian"
        avatar = "🙋" if is_user else "🎓"
        name = "YOU" if is_user else "ADRIAN"
        sources_html = ""
        if not is_user and msg.get("sources"):
            src_tags = "".join(f'<span class="source-badge">📄 p.{s}</span>' for s in msg["sources"])
            sources_html = f'<div style="margin-top:10px;">{src_tags}</div>'
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f'<div class="{row_cls}"><div class="{ava_cls}">{avatar}</div><div><div class="{name_cls}">{name}</div><div class="{bub_cls}">{content}{sources_html}</div></div></div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask Adrian anything from your notes… 💬")
    if user_input and user_input.strip():
        st.session_state.messages.append({"role":"user","content":user_input})
        st.session_state.total_questions += 1
        with st.spinner("Adrian is thinking… 💚"):
            try:
                result = st.session_state.chain({"question": user_input})
                answer = result.get("answer","Hmm, I don't see that in your notes. Try rephrasing? 🤔")
                src_docs = result.get("source_documents",[])
                pages = sorted(set(d.metadata.get("page",0)+1 for d in src_docs if d.metadata.get("page") is not None))[:3]
                st.session_state.messages.append({"role":"assistant","content":answer,"sources":pages})
            except Exception as e:
                st.session_state.messages.append({"role":"assistant","content":f"Oops: `{str(e)[:120]}`\n\nCheck your API key and try again? 🙏"})
        st.rerun()
