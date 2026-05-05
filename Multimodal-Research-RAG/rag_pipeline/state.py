from typing import Optional
from typing_extensions import TypedDict


class RAGState(TypedDict):
    query       : str           # raw user text query
    image_path  : Optional[str] # path to uploaded image, None if text-only

    # ── Query understanding (NEW) ──────────────────────────
    intent           : str      # RETRIEVAL / MEMORY / HYBRID / CHITCHAT
    rewritten_query  : str      # self-contained query for retrieval
    hyde_doc         : str      # hypothetical document for embedding


    # ── Routing (written by query_router) ─────────────────
    query_type  : str           # "text" or "image"

    rewritten_query : str       # self-contained query for retrieval

    
    text_results  : list        # top 20 from text_vector search
    image_results : list        # top 20 from image_vector search

    # Each item in results looks like:
    # {
    #   "score"  : 0.87,
    #   "payload": {
    #     "paper_title": "Attention Is All You Need",
    #     "text"       : "The encoder maps...",
    #     "arxiv_id"   : "1706.03762",
    #     "section"    : "Model Architecture",
    #     "record_type": "text",
    #     "image_path" : None,
    #     "caption"    : "",
    #   }
    # }

    # ── Reranking (written by reranker) ───────────────────
    ranked_results : list       # deduplicated + cross-encoder reranked top 5

    # ── Context building (written by context_builder) ─────
    context  : str              # formatted context string passed to LLM
                                # format: "[1] Paper: X\n    Section: Y\n    text..."
    figures  : list             # figure records to display in UI
                                # [{image_path, caption, paper_title, arxiv_id}]
    sources  : list             # cited papers for display
                                # [{paper_title, arxiv_id, category, url}]

    # ── Generation (written by generator) ─────────────────
    answer   : str              # final LLM response text
    chat_history : list         # conversation history across turns

    # ── Structured conversation state (NEW) ───────────────
    episodic_summary  : str     # compressed summary of older turns
    topics_discussed  : list    # ["transformer architecture", "attention"]
    papers_referenced : list    # ["1706.03762", "2001.04451"]