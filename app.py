"""
Vowel AI Code Assistant (Clean Version)
Run: streamlit run app.py
"""

import streamlit as st
from datetime import datetime
import re as _re

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vowel",
    page_icon="⬡",
    layout="wide"
)

# ── BASIC DARK UI ───────────────────────────────────────────
st.markdown("""
<style>
body { background:#0b0d11; color:#e8eaf2; }
.chat { max-width:900px; margin:auto; }

.user {
    background:#4b4199;
    padding:12px 16px;
    border-radius:12px;
    margin:10px 0;
}

.bot {
    background:#161920;
    padding:12px 16px;
    border-radius:12px;
    margin:10px 0;
    font-family: monospace;
}

/* CODE BLOCK */
pre {
    background:#0b0d11;
    border:1px solid #1f2330;
    border-radius:10px;
    padding:14px;
    overflow-x:auto;
    white-space:pre;
    line-height:1.6;
}

code {
    color:#e8eaf2;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ───────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = []

if "loaded" not in st.session_state:
    st.session_state.loaded = False

# ── LOAD SYSTEM ─────────────────────────────────────────────
@st.cache_resource
def load_system():
    from core import Embedder, KnowledgeBase
    from memory import MemoryManager

    embedder = Embedder()
    kb = KnowledgeBase(embedder)
    memory = MemoryManager(embed_fn=embedder)

    return embedder, kb, memory

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.title("⬡ Vowel")

    if not st.session_state.loaded:
        if st.button("Initialize"):
            with st.spinner("Loading..."):
                embedder, kb, memory = load_system()
                st.session_state.embedder = embedder
                st.session_state.kb = kb
                st.session_state.memory = memory
                st.session_state.loaded = True
                st.rerun()
    else:
        st.success("System Ready")

        if st.button("Clear Chat"):
            st.session_state.chat = []
            st.session_state.memory.clear_all()
            st.rerun()

# ── STOP IF NOT LOADED ─────────────────────────────────────
if not st.session_state.loaded:
    st.info("Initialize system from sidebar")
    st.stop()

# ── RENDER FUNCTION (FIXED CORE) ───────────────────────────
def render_message(text):
    """
    Properly renders:
    - Code blocks (preserved formatting)
    - Inline code
    - Normal text with line breaks
    """

    def process(text):
        parts = []
        last = 0

        for match in _re.finditer(r"```(\w*)\n?(.*?)```", text, _re.DOTALL):
            start, end = match.span()

            # normal text
            normal = text[last:start]
            normal = (
                normal.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            parts.append(normal)

            # code block
            code = match.group(2)
            code = (
                code.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            parts.append(f"<pre><code>{code}</code></pre>")

            last = end

        # remaining text
        remaining = text[last:]
        remaining = (
            remaining.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        parts.append(remaining)

        return "".join(parts)

    html = process(text)

    # inline code
    html = _re.sub(
        r"`([^`]+)`",
        r'<code style="background:#111;padding:2px 5px;border-radius:4px;">\1</code>',
        html
    )

    return html

# ── CHAT DISPLAY ───────────────────────────────────────────
st.markdown('<div class="chat">', unsafe_allow_html=True)

for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f'<div class="user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        rendered = render_message(msg["content"])
        st.markdown(f'<div class="bot">{rendered}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── INPUT ──────────────────────────────────────────────────
user_input = st.text_area("Ask something")
code_input = st.text_area("Code (optional)")

if st.button("Send"):
    if not user_input and not code_input:
        st.warning("Enter something")
        st.stop()

    full_query = user_input
    if code_input:
        full_query += f"\n\n```python\n{code_input}\n```"

    # Save user
    st.session_state.chat.append({
        "role": "user",
        "content": full_query
    })

    st.session_state.memory.add("user", full_query)

    # Retrieve
    kb = st.session_state.kb
    mem = st.session_state.memory

    snippets = kb.retrieve(full_query, top_k=3)

    from core import build_prompt, generate_via_api

    prompt = build_prompt(
        query=full_query,
        snippets=snippets,
        short_term=mem.format_short_term(),
        long_term=mem.format_long_term(full_query),
        mode="answer"
    )

    with st.spinner("Thinking..."):
        answer = generate_via_api(prompt)

    # Clean answer
    answer = _re.sub(r"<think>.*?</think>", "", answer, flags=_re.DOTALL).strip()

    # Save assistant
    st.session_state.chat.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.memory.add("assistant", answer)

    st.rerun()
