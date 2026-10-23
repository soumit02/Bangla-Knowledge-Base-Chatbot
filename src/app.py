# app.py
"""
Step 5: Streamlit web interface for the Bangla Book RAG chatbot.

Run from the PROJECT ROOT (the folder that contains 'src' and '.streamlit'):
    streamlit run src/app.py

Features:
  - Ask a question in Bangla and get an answer based ONLY on the book
  - Chapter citations shown as clickable chips (link to the Wikisource chapter)
  - "Retrieved passages" panel so you can verify every answer against the book
  - Clear message when the answer is not in the book
  - Sample questions, friendly error messages, and a mobile-friendly layout
"""
import html
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st

# ----------------------------------------------------------------------------
# Page setup (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="মেজদিদি — বই-সহায়ক চ্যাটবট",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Book information (keep in sync with ingest.py)
# ----------------------------------------------------------------------------
BOOK_TITLE = "মেজদিদি"
BOOK_AUTHOR = "শরৎচন্দ্র চট্টোপাধ্যায়"
BOOK_URL = "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF"

MAX_QUESTION_CHARS = 500

SAMPLE_QUESTIONS = [
    "‘মেজদিদি’ গল্পটির লেখক কে?",
    "হেমাঙ্গিনীর ছেলের নাম কী?",
    "কেষ্টকে একবার খেতে বসে কী কী খাবার খেতে দেখা গিয়েছিল?",
    "কাদম্বিনী কেষ্টের সঙ্গে কেমন আচরণ করতেন?",
    "বাংলাদেশের রাজধানী কোথায়?",
]

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@600;700&family=Noto+Sans+Bengali:wght@400;500;600&display=swap');

/* Bengali-friendly fonts (applied to text elements only, so Streamlit icons keep working) */
.stMarkdown, .stMarkdown p, .stMarkdown li, .hero, .hero *, .card, .card *,
.chip, .chips-label, .passage, .passage *, .side-box, .side-box *,
[data-testid="stChatInput"] textarea, [data-testid="stExpander"] summary p,
.stButton button p, [data-testid="stSidebar"] .stMarkdown *, .footer-note {
    font-family: 'Noto Sans Bengali', 'Hind Siliguri', 'Segoe UI', sans-serif !important;
}
.hero h1 { font-family: 'Noto Serif Bengali', Georgia, serif !important; }

.block-container { padding-top: 1.6rem; max-width: 980px; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #7C2D12 0%, #B4532A 55%, #D98E4A 100%);
    color: #FFF8EE; border-radius: 22px; padding: 28px 32px; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(124, 45, 18, 0.25);
}
.hero h1 { margin: 0 0 4px 0; font-size: 2.1rem; line-height: 1.25; color: #FFF8EE; }
.hero .author { font-size: 1.05rem; opacity: 0.95; margin-bottom: 4px; }
.hero .subtitle { font-size: 0.92rem; opacity: 0.85; margin-bottom: 14px; }
.badge {
    display: inline-block; background: rgba(255, 248, 238, 0.18);
    border: 1px solid rgba(255, 248, 238, 0.35); color: #FFF8EE;
    border-radius: 999px; padding: 4px 12px; margin: 0 8px 6px 0; font-size: 0.82rem;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #FFFFFF; border: 1px solid #EADBC8; border-radius: 18px;
    padding: 14px 18px; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(43, 33, 24, 0.06);
}
[data-testid="stChatMessage"] p { font-size: 1.04rem; line-height: 1.85; }

/* Cards (not found / error) */
.card { border-radius: 14px; padding: 14px 16px; line-height: 1.75; }
.card-title { font-weight: 600; margin-bottom: 4px; }
.card-warn { background: #FFF4E0; border: 1px solid #F2C879; color: #6B4A10; }
.card-error { background: #FDECEA; border: 1px solid #F1A9A0; color: #7A1F16; }

/* Citation chips */
.chips { margin-top: 10px; }
.chips-label { font-size: 0.85rem; color: #8A6B50; margin-right: 6px; }
a.chip {
    display: inline-block; background: #FDF1E3; border: 1px solid #E8C9A5; color: #8A3B16 !important;
    border-radius: 999px; padding: 3px 12px; margin: 3px 6px 3px 0; font-size: 0.86rem;
    text-decoration: none !important; transition: all 0.15s ease;
}
a.chip:hover { background: #B4532A; color: #FFF8EE !important; border-color: #B4532A; }

/* Retrieved passages */
.passage {
    background: #FFF8EE; border-left: 4px solid #D98E4A; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 10px;
}
.passage-head { font-weight: 600; color: #8A3B16; font-size: 0.88rem; margin-bottom: 4px; }
.passage-text { white-space: pre-wrap; line-height: 1.8; font-size: 0.95rem; color: #3B2D22; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid #EADBC8; }
.side-box {
    background: #FFFFFF; border: 1px solid #EADBC8; border-radius: 14px;
    padding: 12px 14px; margin-bottom: 14px; font-size: 0.92rem; line-height: 1.7;
}
.side-title { font-weight: 600; color: #8A3B16; margin-bottom: 4px; }

/* Buttons */
.stButton button {
    border-radius: 12px; border: 1px solid #E8C9A5; text-align: left; justify-content: flex-start;
    background: #FFFFFF; transition: all 0.15s ease;
}
.stButton button:hover { border-color: #B4532A; background: #FDF1E3; }

.footer-note { text-align: center; color: #9A8470; font-size: 0.8rem; margin-top: 18px; }
#MainMenu, footer { visibility: hidden; }
</style>
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="📚 মডেল ও বইয়ের ডেটাবেস লোড হচ্ছে... (প্রথমবার একটু সময় লাগতে পারে)")
def load_chain():
    """Builds the RAG pipeline once and reuses it for every question."""
    from rag_chain import build_rag_chain, model_name  # imported here so errors can be shown nicely
    return build_rag_chain(), model_name


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)


def queue_question(question: str):
    """Called when a sample-question button is clicked."""
    st.session_state.pending_question = question


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending_question = None


def to_message(result: dict) -> dict:
    """Converts the RAG result into a message that can be stored and rendered later."""
    passages = [
        {"chapter": d.metadata.get("chapter_name", "—"), "text": d.page_content}
        for d in result.get("retrieved_docs", [])
    ]
    return {
        "role": "assistant",
        "content": result["answer"],
        "found": result["found"],
        "error": result.get("error", False),
        "citations": result["citations"],
        "passages": passages,
    }


def chapter_chip(citation: dict) -> str:
    label = html.escape(citation["chapter_name"])
    url = html.escape(citation["source_url"], quote=True)
    return f'<a class="chip" href="{url}" target="_blank" rel="noopener">📘 {label}</a>'


def render_assistant(msg: dict):
    """Shows an assistant message: the answer, citation chips and retrieved passages."""
    if msg.get("error"):
        st.markdown(
            f'<div class="card card-error"><div class="card-title">⚠️ উত্তর তৈরি করা যায়নি</div>'
            f'{html.escape(msg["content"])}</div>',
            unsafe_allow_html=True,
        )
        return

    if not msg["found"]:
        st.markdown(
            '<div class="card card-warn"><div class="card-title">🔍 বইয়ে উত্তর পাওয়া যায়নি</div>'
            f'{html.escape(msg["content"])}<br>'
            '<small>এই চ্যাটবট শুধু ‘মেজদিদি’ বইয়ের তথ্য থেকে উত্তর দেয়, তাই বইয়ের বাইরের কিছু বানিয়ে বলে না।</small>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(msg["content"])
        if msg["citations"]:
            chips = "".join(chapter_chip(c) for c in msg["citations"])
            st.markdown(
                f'<div class="chips"><span class="chips-label">📚 সূত্র (বই থেকে পাওয়া পরিচ্ছেদ):</span>{chips}</div>',
                unsafe_allow_html=True,
            )

    passages = msg.get("passages", [])
    if passages:
        with st.expander(f"📖 বই থেকে খুঁজে পাওয়া অংশ ({len(passages)}টি) — যাচাই করে দেখুন"):
            for i, p in enumerate(passages, start=1):
                st.markdown(
                    f'<div class="passage"><div class="passage-head">{i}. {html.escape(p["chapter"])}</div>'
                    f'<div class="passage-text">{html.escape(p["text"])}</div></div>',
                    unsafe_allow_html=True,
                )


def render_sample_buttons(prefix: str):
    for i, q in enumerate(SAMPLE_QUESTIONS):
        st.button(q, key=f"{prefix}_{i}", on_click=queue_question, args=(q,), use_container_width=True)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
init_state()

# Load the pipeline (shows a friendly message if something is not set up yet)
try:
    ask_book, llm_name = load_chain()
except FileNotFoundError:
    st.error("📂 বইয়ের ডেটাবেস পাওয়া যায়নি। আগে টার্মিনালে `cd src` এবং `python ingest.py` চালান।")
    st.stop()
except ValueError as e:
    st.error(f"⚙️ কনফিগারেশন সমস্যা: {e}")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"❌ চ্যাটবট চালু করা যায়নি: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 📖 বই পরিচিতি")
    st.markdown(
        f'<div class="side-box"><div class="side-title">{BOOK_TITLE}</div>'
        f"লেখক: {BOOK_AUTHOR}<br>"
        f'<a href="{BOOK_URL}" target="_blank" rel="noopener">বাংলা উইকিসংকলনে বইটি পড়ুন ↗</a></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-box"><div class="side-title">কীভাবে কাজ করে?</div>'
        "আপনার প্রশ্ন → বইয়ে প্রাসঙ্গিক অংশ খোঁজা → সেই অংশ থেকে উত্তর ও পরিচ্ছেদের সূত্র।<br>"
        "প্রতিটি প্রশ্নের উত্তর আলাদাভাবে, শুধু বই থেকে দেওয়া হয়।</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### 💡 নমুনা প্রশ্ন")
    render_sample_buttons("side")
    st.button("🗑️ কথোপকথন মুছুন", on_click=clear_chat, use_container_width=True)
    st.caption(f"🤖 LLM: {llm_name}")
    st.caption("🔎 Embedding: BAAI/bge-m3 • Vector DB: ChromaDB")

# Hero banner
st.markdown(
    f"""
    <div class="hero">
        <h1>📖 {BOOK_TITLE}</h1>
        <div class="author">{BOOK_AUTHOR}</div>
        <div class="subtitle">Bangla Book Chatbot — বইটি সম্পর্কে বাংলায় প্রশ্ন করুন</div>
        <span class="badge">✅ শুধু বই থেকে উত্তর</span>
        <span class="badge">📚 প্রতিটি উত্তরে পরিচ্ছেদের সূত্র</span>
        <span class="badge">🚫 উত্তর না থাকলে স্পষ্ট জানিয়ে দেয়</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input (chat_input is always pinned to the bottom of the page)
typed_question = st.chat_input("বইটি সম্পর্কে আপনার প্রশ্ন বাংলায় লিখুন...")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    question = question.strip()[:MAX_QUESTION_CHARS]

# Welcome screen (only before the first question)
if not st.session_state.messages and not question:
    st.markdown("#### 👋 স্বাগতম! নিচের যেকোনো প্রশ্নে ক্লিক করুন, অথবা নিজের প্রশ্ন লিখুন")
    render_sample_buttons("main")

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🙋"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="📖"):
            render_assistant(msg)

# New question
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📖"):
        try:
            with st.spinner("🔎 বইয়ে খোঁজা হচ্ছে..."):
                result = ask_book({"question": question})
            answer_msg = to_message(result)
        except Exception as e:  # noqa: BLE001  (e.g. API limit reached, network problem)
            if "ratelimit" in type(e).__name__.lower() or "429" in str(e):
                friendly = ("LLM-এর API-র ব্যবহারের সীমা (rate limit) ছুঁয়ে গেছে। "
                            "এক-দুই মিনিট অপেক্ষা করে একই প্রশ্ন আবার করুন। "
                            "দিনের সীমা শেষ হলে অন্য মডেল বেছে নিতে হতে পারে।")
            else:
                friendly = (f"কিছু একটা সমস্যা হয়েছে ({type(e).__name__})। "
                            "ইন্টারনেট সংযোগ ও API key যাচাই করে আবার চেষ্টা করুন।")
            answer_msg = {
                "role": "assistant",
                "content": friendly,
                "found": False,
                "error": True,
                "citations": [],
                "passages": [],
            }
        render_assistant(answer_msg)

    st.session_state.messages.append(answer_msg)

st.markdown(
    '<div class="footer-note">AI ভুল করতে পারে — গুরুত্বপূর্ণ তথ্য সূত্রের পরিচ্ছেদ খুলে যাচাই করে নিন।</div>',
    unsafe_allow_html=True,
)