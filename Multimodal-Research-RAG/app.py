"""
app.py — Streamlit UI for Multimodal RAG Pipeline
Run: streamlit run app.py
"""

import os
import sys
import base64
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon ="🔬",
    layout    ="centered",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline.nodes import ConversationMemory
from rag_pipeline.run   import run_query


# ============================================================
# Helpers
# ============================================================

def img_to_base64(image_path: str):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_figure(fig: dict):
    img_path = fig.get("image_path", "")
    caption  = fig.get("caption", "")
    if caption.startswith("Figure from"):
        caption = ""
    b64 = img_to_base64(img_path)
    if b64:
        st.image(
            f"data:image/png;base64,{b64}",
            caption         =caption if caption else None,
            use_column_width=True,
        )
    else:
        st.caption(f"📊 {caption or 'Figure'}")


def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)} papers)", expanded=False):
        for s in sources:
            title    = s.get("paper_title", "Unknown")
            arxiv_id = s.get("arxiv_id", "")
            category = s.get("category", "")
            url      = s.get("url", f"https://arxiv.org/abs/{arxiv_id}")
            st.markdown(
                f"**{title}**  \n"
                f"[{arxiv_id}]({url}) · {category}"
            )


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "raw_client" not in st.session_state:
        st.session_state.raw_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    #MainMenu  {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}
    .main .block-container {
        max-width  : 800px;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 AI Research Assistant")
        st.markdown(
            "Ask questions about **200+ AI research papers** "
            "across 7 categories."
        )
        st.divider()

        st.markdown("**Categories:**")
        for cat in [
            "🤖 Transformers & Attention",
            "📈 LLMs & Scaling",
            "🔍 RAG & Retrieval",
            "⚡ Efficient & Fine-tuning",
            "🎯 RL & Alignment",
            "🖼️ CV & Multimodal",
            "🎨 Diffusion & Generation",
        ]:
            st.markdown(f"  {cat}")

        st.divider()

        memory = st.session_state.get("memory")
        if memory:
            st.markdown("**Conversation:**")
            st.markdown(f"  Turns: `{memory.total_turns}`")
            st.markdown(f"  Topics: `{len(memory.topics_discussed)}`")
            summary_status = "✅ active" if memory.episodic_summary else "—"
            st.markdown(f"  Summary: `{summary_status}`")

        st.divider()

        st.markdown("**Try asking:**")
        for q in [
            "How does attention work?",
            "What is RAG?",
            "Compare Mamba vs Transformer",
            "Show me the transformer architecture",
            "How does RLHF work?",
            "What are scaling laws?",
        ]:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.prefill_query = q
                st.rerun()

        st.divider()

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.messages = []
            st.session_state.memory   = ConversationMemory()
            st.rerun()


# ============================================================
# Message rendering
# ============================================================

def render_message(msg: dict):
    role = msg["role"]
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
            # show attached image if any
            if msg.get("image_b64"):
                st.image(
                    f"data:image/png;base64,{msg['image_b64']}",
                    width=150,
                )
    else:
        with st.chat_message("assistant", avatar="🔬"):
            st.markdown(msg["content"])
            figures = msg.get("figures", [])
            if figures:
                st.markdown("---")
                cols_per_row = 2
                for i in range(0, len(figures), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, fig in enumerate(figures[i:i+cols_per_row]):
                        with cols[j]:
                            render_figure(fig)
            render_sources(msg.get("sources", []))


# ============================================================
# Query processing
# ============================================================

def process_query(query: str, image_path: str = None):
    memory     = st.session_state.memory
    raw_client = st.session_state.raw_client

    if memory.should_summarize():
        memory.summarize(raw_client)

    result = run_query(
        query        = query,
        image_path   = image_path,
        chat_history = memory.to_messages(),
        extra_state  = {
            "episodic_summary" : memory.episodic_summary,
            "topics_discussed" : memory.topics_discussed,
            "papers_referenced": memory.papers_referenced,
        }
    )

    memory.add_turn(
        user_msg      = query or f"[image: {image_path}]",
        assistant_msg = result["answer"],
        topics        = result.get("topics_discussed", []),
        papers        = result.get("papers_referenced", []),
    )

    if memory.should_summarize():
        memory.summarize(raw_client)

    return result


# ============================================================
# Main
# ============================================================

def main():
    initialize_session()
    render_sidebar()

    # header
    st.markdown(
        "<h2 style='text-align:center; color:#90cdf4;'>"
        "🔬 AI Research Assistant"
        "</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#718096; font-size:0.9rem;'>"
        "Ask anything about AI research papers"
        "</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # render existing messages
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🔬"):
            st.markdown(
                "👋 Hi! I'm your AI research assistant. I have access to "
                "**200+ AI research papers** across transformers, LLMs, "
                "RAG, diffusion models, and more.\n\n"
                "Ask me anything — I'll answer from the papers and show "
                "relevant figures when available."
            )
    else:
        for msg in st.session_state.messages:
            render_message(msg)

    # handle prefilled query from sidebar
    prefill = st.session_state.pop("prefill_query", None)

    # ── chat input with native file upload ────────────────
    uploaded_image_path = None
    query               = ""

    try:
        # streamlit >= 1.31.0 supports accept_file
        prompt = st.chat_input(
            "Ask about any AI research topic...",
            accept_file=True,
            file_type  =["png", "jpg", "jpeg"],
        )

        if prompt:
            query         = prompt.text or ""
            uploaded_file = prompt.files[0] if prompt.files else None

            if uploaded_file:
                temp_path = Path("temp_upload.png")
                temp_path.write_bytes(uploaded_file.getvalue())
                uploaded_image_path = str(temp_path)

    except TypeError:
        # fallback for older streamlit — no accept_file support
        prompt = st.chat_input(
            "Ask about any AI research topic...",
        )
        if prompt:
            query = prompt

    # handle prefill from sidebar buttons
    if prefill and not query:
        query = prefill

    # process query
    if query or uploaded_image_path:

        # store image as base64 for re-rendering in history
        image_b64 = None
        if uploaded_image_path:
            image_b64 = img_to_base64(uploaded_image_path)

        # add user message
        st.session_state.messages.append({
            "role"     : "user",
            "content"  : query or "🖼️ Image query",
            "image_b64": image_b64,
        })

        # render user message
        with st.chat_message("user", avatar="👤"):
            if query:
                st.markdown(query)
            if image_b64:
                st.image(
                    f"data:image/png;base64,{image_b64}",
                    width=150,
                )

        # run pipeline and render response
        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("Searching research papers..."):
                try:
                    result  = process_query(query, uploaded_image_path)
                    answer  = result["answer"]
                    figures = result.get("figures", [])
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if figures:
                        st.markdown("---")
                        cols_per_row = 2
                        for i in range(0, len(figures), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, fig in enumerate(
                                figures[i:i+cols_per_row]
                            ):
                                with cols[j]:
                                    render_figure(fig)

                    render_sources(sources)

                    st.session_state.messages.append({
                        "role"   : "assistant",
                        "content": answer,
                        "figures": figures,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role"   : "assistant",
                        "content": error_msg,
                        "figures": [],
                        "sources": [],
                    })

    # cleanup temp image
    if uploaded_image_path and Path(uploaded_image_path).exists():
        try:
            Path(uploaded_image_path).unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()