"""
app.py — Vowel AI Code Assistant with persistent memory.
Run: streamlit run app.py 
"""

import streamlit as st
from datetime import datetime

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vowel",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

/* ── ROOT TOKENS ── */
:root {
  --bg:          #0b0d11;
  --surface:     #111318;
  --panel:       #161920;
  --border:      #1f2330;
  --border-lit:  #2e3448;
  --accent:      #7c6af7;
  --accent-dim:  #4b4199;
  --accent-glow: #7c6af730;
  --green:       #4ade80;
  --green-dim:   #166534;
  --amber:       #fbbf24;
  --red:         #f87171;
  --text-1:      #e8eaf2;
  --text-2:      #8b90a8;
  --text-3:      #4a4f66;
  --radius:      10px;
  --radius-lg:   16px;
}

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text-1) !important;
  font-family: 'Syne', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── MAIN COLUMN PADDING ── */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 99px; }

/* ── TYPOGRAPHY ── */
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* ── BRAND HEADER ── */
.brand-header { padding: 28px 24px 20px; border-bottom: 1px solid var(--border); }
.brand-hex {
  width: 36px; height: 36px;
  background: var(--accent);
  clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
  margin-bottom: 12px;
  box-shadow: 0 0 20px var(--accent-glow);
}
.brand-name { font-size: 20px; font-weight: 800; color: var(--text-1); letter-spacing: -0.03em; line-height: 1; }
.brand-sub { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--text-3); margin-top: 4px; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── SIDEBAR SECTION LABELS ── */
.sb-label { font-size: 10px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3); padding: 20px 24px 8px; }

/* ── STATUS BADGE ── */
.status-badge { display: flex; align-items: center; gap: 8px; padding: 10px 24px; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot.online  { background: var(--green); box-shadow: 0 0 8px var(--green); }
.status-dot.offline { background: var(--text-3); }
.status-dot.pulse   { background: var(--amber); box-shadow: 0 0 8px var(--amber); animation: pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:.5;transform:scale(.8);} }

/* ── STAT ROW ── */
.stat-row { display: flex; gap: 1px; margin: 0 24px 16px; }
.stat-card { flex: 1; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 12px; text-align: center; }
.stat-num { font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--accent); line-height: 1; }
.stat-label { font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-3); margin-top: 4px; }

/* ── HISTORY ITEMS ── */
.hist-item { margin: 0 16px 6px; padding: 10px 12px; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); cursor: pointer; transition: border-color .15s; }
.hist-item:hover { border-color: var(--border-lit); }
.hist-ts { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: var(--text-3); margin-bottom: 4px; }
.hist-preview { font-size: 12px; color: var(--text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-preview strong { color: var(--accent); font-weight: 600; }

/* ── MAIN AREA ── */
.main-wrap { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* ── TOP BAR ── */
.topbar { display: flex; align-items: center; justify-content: space-between; padding: 16px 32px; border-bottom: 1px solid var(--border); background: var(--surface); flex-shrink: 0; }
.topbar-title { font-size: 15px; font-weight: 700; color: var(--text-1); display: flex; align-items: center; gap: 10px; }
.topbar-mode { font-size: 11px; font-family: 'JetBrains Mono', monospace; background: var(--accent-glow); color: var(--accent); border: 1px solid var(--accent-dim); padding: 3px 10px; border-radius: 99px; }
.topbar-meta { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--text-3); }

/* ── CHAT AREA ── */
.chat-feed { flex: 1; overflow-y: auto; padding: 32px; display: flex; flex-direction: column; gap: 24px; }

/* ── EMPTY STATE ── */
.empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; color: var(--text-3); padding: 60px 40px; }
.empty-hex { width: 56px; height: 56px; background: var(--panel); border: 1px solid var(--border); clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%); display: flex; align-items:center; justify-content:center; font-size: 22px; }
.empty-title { font-size: 16px; font-weight: 700; color: var(--text-2); }
.empty-body { font-size: 13px; text-align: center; max-width: 360px; line-height: 1.7; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px; }
.chip { background: var(--panel); border: 1px solid var(--border); border-radius: 99px; padding: 6px 14px; font-size: 12px; font-family: 'JetBrains Mono', monospace; color: var(--text-2); }

/* ── MESSAGE BUBBLES ── */
.msg-row { display: flex; flex-direction: column; max-width: 840px; }
.msg-row.user { align-self: flex-end; align-items: flex-end; }
.msg-row.bot  { align-self: flex-start; align-items: flex-start; }
.msg-meta { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: var(--text-3); margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
.msg-meta .who { font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.msg-row.user .who { color: var(--accent); }
.msg-row.bot  .who { color: var(--green); }
.bubble { padding: 14px 18px; border-radius: var(--radius-lg); font-size: 14px; line-height: 1.75; max-width: 680px; word-break: break-word; }
.bubble.user { background: var(--accent-dim); border: 1px solid var(--accent); color: #ddd8ff; border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg); }
.bubble.bot  { background: var(--panel); border: 1px solid var(--border); color: var(--text-1); border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px; font-family: 'JetBrains Mono', monospace; font-size: 13px; white-space: pre-wrap; }
.bubble.bot code { background: var(--bg); padding: 2px 6px; border-radius: 4px; font-size: 12px; color: var(--amber); }

/* ── MEMORY NOTICE ── */
.mem-notice { display: flex; align-items: center; gap: 6px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--accent); margin-top: 6px; opacity: .75; }
.mem-dot { width: 5px; height: 5px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 6px var(--accent); }

/* ── INPUT ZONE ── */
.input-zone { padding: 20px 32px 28px; background: var(--surface); border-top: 1px solid var(--border); flex-shrink: 0; }
.input-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.input-label { font-size: 10px; font-family: 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--text-3); margin-bottom: 6px; }
.send-row { display: flex; align-items: center; justify-content: space-between; }
.send-hint { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--text-3); }

/* ── STREAMLIT INPUT OVERRIDES ── */
[data-testid="stTextArea"] textarea { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; color: var(--text-1) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; resize: none !important; }
[data-testid="stTextArea"] textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-glow) !important; }
[data-testid="stTextArea"] textarea::placeholder { color: var(--text-3) !important; }
[data-testid="stTextArea"] label { display: none !important; }

/* ── STREAMLIT SELECT ── */
[data-testid="stSelectbox"] > div > div { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; color: var(--text-1) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }
[data-testid="stSelectbox"] label { color: var(--text-2) !important; font-size: 12px !important; }

/* ── STREAMLIT BUTTONS ── */
[data-testid="stButton"] button { background: var(--panel) !important; border: 1px solid var(--border) !important; color: var(--text-2) !important; border-radius: var(--radius) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; transition: all .15s !important; }
[data-testid="stButton"] button:hover { border-color: var(--border-lit) !important; color: var(--text-1) !important; }
[data-testid="stButton"] button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; font-weight: 600 !important; box-shadow: 0 0 20px var(--accent-glow) !important; }
[data-testid="stButton"] button[kind="primary"]:hover { background: #6b5ce7 !important; box-shadow: 0 0 30px var(--accent-glow) !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; margin: 8px 0 !important; }

/* ── BOOT SCREEN ── */
.boot-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh; gap: 20px; color: var(--text-3); }
.boot-hex { width: 72px; height: 72px; background: var(--accent); clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%); box-shadow: 0 0 40px var(--accent-glow); animation: float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-8px);} }
.boot-title { font-size: 28px; font-weight: 800; color: var(--text-1); letter-spacing: -.03em; }
.boot-body { font-size: 13px; font-family: 'JetBrains Mono', monospace; text-align: center; line-height: 1.8; max-width: 420px; }
.boot-steps { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px 24px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-2); line-height: 2; min-width: 320px; }
.boot-steps span { color: var(--accent); margin-right: 8px; }

/* ── SIDEBAR DIVIDER ── */
.sb-divider { height: 1px; background: var(--border); margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "loaded":     False,
        "embedder":   None,
        "kb":         None,
        "memory":     None,
        "chat":       [],
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode":       "answer",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── CACHE LOADER ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    from core import Embedder, KnowledgeBase
    embedder = Embedder()
    kb = KnowledgeBase(embedder)
    return embedder, kb

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-header">
      <div class="brand-hex"></div>
      <div class="brand-name">Vowel</div>
      <div class="brand-sub">AI · Code · Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    if not st.session_state.loaded:
        st.markdown("""
        <div class="status-badge">
          <div class="status-dot offline"></div>
          <span style="color:var(--text-3);font-family:'JetBrains Mono',monospace;font-size:12px;">System offline</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Startup</div>', unsafe_allow_html=True)
        if st.button("⬡ Initialize System", use_container_width=True, type="primary"):
            with st.spinner("Booting Vowel…"):
                from core import Embedder, KnowledgeBase
                from memory import MemoryManager
                embedder, kb = load_system()
                memory = MemoryManager(
                    embed_fn=embedder,
                    session_id=st.session_state.session_id
                )
                st.session_state.embedder = embedder
                st.session_state.kb       = kb
                st.session_state.memory   = memory
                st.session_state.loaded   = True
                st.rerun()
    else:
        st.markdown("""
        <div class="status-badge">
          <div class="status-dot online"></div>
          <span style="color:var(--green);font-family:'JetBrains Mono',monospace;font-size:12px;">System ready</span>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        mem   = st.session_state.memory
        total = len(mem.get_all_history())
        vecs  = mem.index.ntotal
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="stat-num">{total}</div><div class="stat-label">Messages</div></div>
          <div class="stat-card"><div class="stat-num">{vecs}</div><div class="stat-label">Vectors</div></div>
          <div class="stat-card"><div class="stat-num">{len(st.session_state.chat)//2}</div><div class="stat-label">Session</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Mode
        st.markdown('<div class="sb-label">Mode</div>', unsafe_allow_html=True)
        mode = st.selectbox(
            "mode",
            ["answer", "explain", "fix", "improve"],
            format_func=lambda x: {
                "answer":  "💬 Q&A",
                "explain": "📖 Explain",
                "fix":     "🔧 Fix Bug",
                "improve": "✨ Improve",
            }[x],
            label_visibility="collapsed",
        )
        st.session_state.mode = mode

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Memory controls
        st.markdown('<div class="sb-label">Memory</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("↩ Undo last", use_container_width=True):
                mem.delete_last()
                if len(st.session_state.chat) >= 2:
                    st.session_state.chat = st.session_state.chat[:-2]
                st.rerun()
        with c2:
            if st.button("✕ Clear all", use_container_width=True):
                mem.clear_all()
                st.session_state.chat = []
                st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # History
        st.markdown('<div class="sb-label">History</div>', unsafe_allow_html=True)
        history    = mem.get_all_history()
        user_turns = [h for h in history if h["role"] == "user"]

        if not user_turns:
            st.markdown(
                '<p style="font-size:12px;color:var(--text-3);padding:0 24px;'
                'font-family:\'JetBrains Mono\',monospace;">No history yet.</p>',
                unsafe_allow_html=True
            )
        else:
            for item in reversed(user_turns[-8:]):
                ts_short = item["ts"][:16].replace("T", " ")
                preview  = item["content"][:55] + ("…" if len(item["content"]) > 55 else "")
                st.markdown(
                    f'<div class="hist-item">'
                    f'  <div class="hist-ts">{ts_short}</div>'
                    f'  <div class="hist-preview"><strong>›</strong> {preview}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ── NOT LOADED SCREEN ──────────────────────────────────────────────────────────
if not st.session_state.loaded:
    st.markdown("""
    <div class="boot-screen">
      <div class="boot-hex"></div>
      <div class="boot-title">Vowel</div>
      <div class="boot-body">
        An AI code assistant with persistent memory.<br>
        Explain · Fix · Improve · Answer — context lives across sessions.
      </div>
      <div class="boot-steps">
        <div><span>01</span>Click <b>Initialize System</b> in the sidebar</div>
        <div><span>02</span>Wait ~30 seconds on first run (model cold-start)</div>
        <div><span>03</span>Ask anything about code</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── TOPBAR ─────────────────────────────────────────────────────────────────────
mode_labels  = {"answer": "Q&A", "explain": "Explain", "fix": "Fix Bug", "improve": "Improve"}
current_mode = st.session_state.get("mode", "answer")
session_ts   = st.session_state.session_id

st.markdown(f"""
<div class="topbar">
  <div class="topbar-title">
    ⬡ &nbsp;Vowel
    <span class="topbar-mode">{mode_labels.get(current_mode, "Q&A")}</span>
  </div>
  <div class="topbar-meta">session · {session_ts}</div>
</div>
""", unsafe_allow_html=True)

# ── CHAT FEED ──────────────────────────────────────────────────────────────────
import re as _re

chat = st.session_state.chat

if not chat:
    st.markdown("""
    <div class="chat-feed">
      <div class="empty-state">
        <div class="empty-hex">⬡</div>
        <div class="empty-title">Ready when you are</div>
        <div class="empty-body">
          Ask me to explain a concept, fix a bug, improve your code,
          or answer any programming question.
        </div>
        <div class="chip-row">
          <span class="chip">explain this function</span>
          <span class="chip">find the bug</span>
          <span class="chip">how does X work?</span>
          <span class="chip">make this faster</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    bubbles_html = '<div class="chat-feed">'
    for msg in chat:
        ts   = msg.get("ts", "")[:16].replace("T", " ")
        role = msg["role"]

        if role == "user":
            content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            bubbles_html += f"""
            <div class="msg-row user">
              <div class="msg-meta"><span class="who">You</span><span>{ts}</span></div>
              <div class="bubble user">{content}</div>
            </div>"""
        else:
            raw = msg["content"]
            # Strip chain-of-thought tags
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL)
            raw = _re.sub(r"<think>.*",          "", raw, flags=_re.DOTALL)
            raw = _re.sub(r"</?think>",           "", raw).strip()

            content = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            def _code_block(m):
                code = m.group(2).replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
                code = code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                return (
                    f'<pre style="background:#0b0d11;border:1px solid #1f2330;'
                    f'border-radius:8px;padding:12px 16px;margin:8px 0;overflow-x:auto;">'
                    f'<code style="color:#e8eaf2;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:12px;line-height:1.6;">{code}</code></pre>'
                )

            content = _re.sub(r"```(\w*)\n?(.*?)```", _code_block, content, flags=_re.DOTALL)
            content = _re.sub(
                r"`([^`]+)`",
                r'<code style="background:#0b0d11;padding:2px 5px;border-radius:4px;'
                r'color:#fbbf24;font-size:12px;">\1</code>',
                content
            )
            content = content.replace("\n", "<br>")

            mem_tag = ""
            if msg.get("used_memory"):
                mem_tag = '<div class="mem-notice"><div class="mem-dot"></div>drawing on our earlier conversation</div>'

            bubbles_html += f"""
            <div class="msg-row bot">
              <div class="msg-meta"><span class="who">Vowel</span><span>{ts}</span></div>
              <div class="bubble bot" style="white-space:normal;">{content}</div>
              {mem_tag}
            </div>"""

    bubbles_html += "</div>"
    st.markdown(bubbles_html, unsafe_allow_html=True)

# ── INPUT ZONE ─────────────────────────────────────────────────────────────────
st.markdown('<div class="input-zone"><div class="input-grid">', unsafe_allow_html=True)

col_q, col_c = st.columns(2)

with col_q:
    st.markdown('<div class="input-label">Your question</div>', unsafe_allow_html=True)
    user_msg = st.text_area(
        "question",
        placeholder="Explain · Fix · Improve · Ask anything…",
        height=110,
        key="user_input",
        label_visibility="collapsed",
    )

with col_c:
    st.markdown('<div class="input-label">Code <span style="opacity:.4">(optional)</span></div>', unsafe_allow_html=True)
    code_input = st.text_area(
        "code",
        placeholder="Paste code here…",
        height=110,
        key="code_input",
        label_visibility="collapsed",
    )

st.markdown('</div>', unsafe_allow_html=True)

col_send, col_hint = st.columns([1, 4])
with col_send:
    send = st.button("Send ▶", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        '<div class="send-hint" style="padding-top:8px;">Vowel · Mistral-7B via HuggingFace Free API · persistent memory</div>',
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── PROCESS QUERY ──────────────────────────────────────────────────────────────
if send and (user_msg.strip() or code_input.strip()):
    from core import build_prompt, generate_via_api

    full_query = user_msg.strip()
    if code_input.strip():
        full_query += f"\n\n```python\n{code_input.strip()}\n```"

    ts_now = datetime.now().isoformat()
    mode   = st.session_state.get("mode", "answer")
    mem    = st.session_state.memory
    kb     = st.session_state.kb

    st.session_state.chat.append({"role": "user", "content": full_query, "ts": ts_now})
    mem.add("user", full_query)

    snippets  = kb.retrieve(full_query, top_k=3)
    short_ctx = mem.format_short_term()
    long_ctx  = mem.format_long_term(full_query)
    used_mem  = mem.used_long_term(full_query)

    prompt = build_prompt(
        query=full_query, snippets=snippets, mode=mode,
        short_term=short_ctx, long_term=long_ctx,
    )

    with st.spinner("⬡ Thinking…"):
        answer = generate_via_api(prompt, max_tokens=512)

    # Final safety strip — catches any think tags that slipped through
    answer = _re.sub(r"<think>.*?</think>", "", answer, flags=_re.DOTALL)
    answer = _re.sub(r"<think>.*",          "", answer, flags=_re.DOTALL)
    answer = _re.sub(r"</?think>",           "", answer).strip()

    if not answer:
        answer = "No response generated. Please rephrase your question."

    ts_ans = datetime.now().isoformat()
    st.session_state.chat.append({
        "role":        "assistant",
        "content":     answer,
        "ts":          ts_ans,
        "used_memory": used_mem,
    })
    mem.add("assistant", answer)
    st.rerun()
