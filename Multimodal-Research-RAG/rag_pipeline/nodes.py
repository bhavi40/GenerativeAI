"""
nodes.py — All LangGraph pipeline nodes

Each node:
  - receives the full RAGState
  - does one job
  - returns a dict of state keys it wants to update

Nodes never call each other directly —
LangGraph handles the wiring via graph.py

Nodes in order:
  0. query_rewriter   ← NEW: rewrites follow-up questions using chat history
  1. query_router     ← decides text or image path
  2A. text_retriever  ← BGE embed + Qdrant text search
  2B. image_retriever ← CLIP embed + Qdrant image search
  3. reranker         ← deduplicate + cross-encoder rerank
  4. context_builder  ← format context for LLM
  5. generator        ← Claude LLM + chat history
"""

import os
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .state import RAGState
from .models import embed_text, embed_image, embed_texts_batch
from .retriever import search_by_text, search_by_image, get_paper_figures
import anthropic
import re
import json
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """
    Two-tier memory system.
    Tier 1 — episodic_summary : compressed older turns
    Tier 2 — recent_turns     : last N turns verbatim
    """
    episodic_summary  : str  = ""
    recent_turns      : list = field(default_factory=list)
    total_turns       : int  = 0
    topics_discussed  : list = field(default_factory=list)
    papers_referenced : list = field(default_factory=list)

    RECENT_TURNS_LIMIT  : int = 6    # keep last 6 turns verbatim
    SUMMARIZE_THRESHOLD : int = 10   # only summarize at exact multiples

    def add_turn(
        self,
        user_msg      : str,
        assistant_msg : str,
        topics        : list = None,
        papers        : list = None,
    ):
        self.recent_turns.append({"role": "user",      "content": user_msg})
        self.recent_turns.append({"role": "assistant", "content": assistant_msg})
        self.total_turns += 1
        if topics:
            for t in topics:
                if t not in self.topics_discussed:
                    self.topics_discussed.append(t)
        if papers:
            for p in papers:
                if p not in self.papers_referenced:
                    self.papers_referenced.append(p)

    def should_summarize(self) -> bool:
        """
        Trigger summarization only at exact multiples of SUMMARIZE_THRESHOLD.
        Turn 10 → True  (first summarization)
        Turn 11 → False
        Turn 20 → True  (second summarization)
        Turn 21 → False
        """
        return (
            self.total_turns > 0
            and self.total_turns % self.SUMMARIZE_THRESHOLD == 0
        )

    def summarize(self, client: anthropic.Anthropic):
        """
        Compress old turns into episodic summary.
        Accepts existing client — no re-initialization.
        """
        keep_recent       = self.RECENT_TURNS_LIMIT * 2
        old_turns         = self.recent_turns[:-keep_recent]
        self.recent_turns = self.recent_turns[-keep_recent:]

        if not old_turns:
            return

        old_text = "\n".join([
            f"{m['role'].upper()}: {m['content'][:400]}"
            for m in old_turns
        ])

        existing = (
            f"Existing summary: {self.episodic_summary}\n\n"
            if self.episodic_summary else ""
        )

        topics_str = ", ".join(self.topics_discussed) if self.topics_discussed else "none yet"

        prompt = (
            f"{existing}"
            f"New turns to fold in:\n{old_text}\n\n"
            f"Topics discussed so far: {topics_str}\n\n"
            f"Write a concise technical summary (4-6 sentences). "
            f"Include: paper names cited, exact technical concepts explained, "
            f"conclusions reached, and any comparisons made. "
            f"Be specific — this replaces the raw conversation history."
        )

        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 400,
            messages   = [{"role": "user", "content": prompt}]
        )
        self.episodic_summary = response.content[0].text
        print(
            f"[memory] turn {self.total_turns} → "
            f"summarized {len(old_turns)//2} old turns into episodic summary"
        )

    def to_messages(self) -> list:
        messages = []
        if self.episodic_summary:
            messages.append({
                "role"   : "user",
                "content": (
                    f"[CONVERSATION CONTEXT — previous session summary]\n"
                    f"{self.episodic_summary}"
                )
            })
            messages.append({
                "role"   : "assistant",
                "content": "Understood. I have the context of our previous discussion."
            })
        messages.extend(self.recent_turns)
        return messages

    def reset(self):
        self.episodic_summary  = ""
        self.recent_turns      = []
        self.total_turns       = 0
        self.topics_discussed  = []
        self.papers_referenced = []


# ── LLM setup ─────────────────────────────────────────────
def get_llm(max_tokens: int = 1500):
    """
    Create Claude LLM instance.
    Not a singleton — each call creates fresh instance.
    LLM calls are stateless API requests so no need to reuse.

    max_tokens controls response length:
      1500 for main answers
      300  for query rewriting (short task)
    """
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=max_tokens,
    )



# ============================================================
# NODE 0 — query_rewriter (3-level query understanding)
# ============================================================


MEMORY_PATTERNS = [
    r"what did you (just |)explain",
    r"what (did|have) we discuss",
    r"summarize (what|our)",
    r"what was the (last|previous)",
    r"can you repeat",
    r"what (just |)said",
    r"what (have )?you told me",
    r"remind me",
    r"go back to",
    r"earlier you said",
    r"you mentioned",
    r"we (just |)talked about",
]


def _classify_intent(
    query        : str,
    chat_history : list,
    client       : anthropic.Anthropic,
) -> str:
    history_text = "\n".join([
        f"{m['role']}: {m['content'][:200]}"
        for m in chat_history[-6:]
    ]) if chat_history else "None"

    response = client.messages.create(
        model      = "claude-haiku-4-5-20251001",
        max_tokens = 15,
        messages   = [{
            "role"   : "user",
            "content": (
                f"Classify this query into exactly one category. "
                f"Reply with ONLY the category name.\n\n"
                f"Categories:\n"
                f"- RETRIEVAL: needs searching research papers for new information\n"
                f"- MEMORY: ONLY about recalling what was already said "
                f"(e.g. 'what did you explain', 'what did we discuss', "
                f"'summarize our conversation', 'what did you just say')\n"
                f"- HYBRID: references previous conversation AND needs new "
                f"information from papers "
                f"(e.g. 'compare X with what we discussed', "
                f"'how does that relate to Y', 'expand on that with Z')\n"
                f"- CHITCHAT: greeting or completely off-topic\n\n"
                f"Key rule: if the query wants NEW information beyond just "
                f"recalling the conversation → HYBRID not MEMORY\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"Query: {query}\n\n"
                f"Category:"
            )
        }]
    )

    intent = response.content[0].text.strip().upper()
    valid  = {"RETRIEVAL", "MEMORY", "HYBRID", "CHITCHAT"}
    intent = intent if intent in valid else "RETRIEVAL"

    # ── Fallback rule: MEMORY + comparison signal → HYBRID ──
    # Classifier struggles when query has both "what we discussed"
    # (memory signal) AND comparison/expansion signals (retrieval)
    comparison_signals = [
        "compare", "versus", "vs", "difference",
        "how does that relate", "expand", "contrast",
        "better than", "worse than", "similar to",
    ]
    if intent == "MEMORY" and any(
        s in query.lower() for s in comparison_signals
    ):
        intent = "HYBRID"
        print(f"[classify] overridden MEMORY → HYBRID (comparison signal detected)")

    return intent

def _extract_entities(
    query        : str,
    chat_history : list,
    llm,                    # LangChain LLM instance
) -> str:
    """
    Level 2 — Entity resolution (Anaphora Resolution).

    Solves the "it" problem:
      "how does it work?"
      → "how does scaled dot-product attention work?"

    Without this, BGE embeds "it" and retrieves garbage.
    With this, BGE has concrete nouns to match against chunks.

    Only runs when chat_history exists — no history means
    no references to resolve.
    """
    if not chat_history:
        return query

    history_text = ""
    for turn in chat_history[-6:]:
        role    = "User"      if turn["role"] == "user" else "Assistant"
        content = turn["content"][:300]
        history_text += f"{role}: {content}\n\n"

    messages = [
        SystemMessage(content=(
            "You are a query rewriter for a RAG search system.\n"
            "Resolve all pronouns and references in the query "
            "using the conversation history.\n"
            "Make the query completely self-contained.\n"
            "Return ONLY the rewritten query, nothing else.\n"
            "If the query is already self-contained, return it unchanged."
        )),
        HumanMessage(content=(
            f"Conversation history:\n{history_text}\n"
            f"Query: {query}\n\n"
            f"Self-contained query:"
        ))
    ]

    response  = llm.invoke(messages)
    rewritten = response.content.strip()
    return rewritten if rewritten and len(rewritten) > 5 else query


def _decompose_query(query: str, llm) -> list[str]:
    """
    Level 2 — Sub-query decomposition.

    Solves the "Compare X vs Y" problem:
      Most embedding models struggle with multi-part queries
      because they search for one chunk containing everything.

    "compare attention in transformers vs mamba"
    → ["attention mechanism in transformers",
       "attention mechanism in mamba",
       "comparison of transformer and mamba architectures"]

    Three separate Qdrant searches → much better coverage.
    Only runs when complexity signals are detected.
    """
    complexity_signals = [
        "compare", "versus", "vs", "difference between",
        "similarities", "both", "all of", "each of",
        "contrast", "relationship between"
    ]
    is_complex = any(s in query.lower() for s in complexity_signals)

    if not is_complex:
        return [query]

    messages = [
        SystemMessage(content=(
            "You are a query decomposer for a research paper search system.\n"
            "Break the query into 2-3 simpler sub-queries for better retrieval.\n"
            "Return ONLY a JSON array of strings, nothing else.\n"
            'Example: ["sub-query 1", "sub-query 2", "sub-query 3"]'
        )),
        HumanMessage(content=f"Query: {query}\n\nSub-queries:")
    ]

    try:
        response = llm.invoke(messages)
        content  = response.content.strip()

        # bulletproof JSON parsing — Claude sometimes wraps in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        sub_queries = json.loads(content)

        if isinstance(sub_queries, list) and len(sub_queries) > 1:
            print(f"[query_rewriter] decomposed into {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries):
                print(f"  [{i+1}] {sq}")
            return sub_queries

    except json.JSONDecodeError as e:
        print(f"[query_rewriter] decomposition JSON parse failed: {e} → using original")
    except Exception as e:
        print(f"[query_rewriter] decomposition failed: {e} → using original")

    return [query]


def _generate_hyde_document(query: str, llm) -> str:
    """
    HyDE — Hypothetical Document Embedding.

    Instead of embedding the raw query, generate a hypothetical
    passage that looks like a research paper chunk answering
    the question. This embeds much closer to actual paper chunks.

    Raw query vector  ←————————→ chunk vectors  (far apart)
    HyDE doc vector   ←→ chunk vectors           (much closer)

    Cost : one small LLM call (~$0.0001 with Haiku)
    Gain : dramatically better retrieval for figures
           and specific technical concepts
    """
    messages = [
        SystemMessage(content=(
            "You are a research paper passage generator.\n"
            "Generate a SHORT passage (2-4 sentences) that would appear "
            "in an AI research paper and directly answers the question.\n"
            "Rules:\n"
            "- Write in academic paper style\n"
            "- Be specific and technical\n"
            "- If asking about a figure/diagram, mention it explicitly\n"
            "- Return ONLY the passage, no explanation"
        )),
        HumanMessage(content=f"Question: {query}\n\nPassage:")
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def query_rewriter(state: RAGState) -> dict:
    """
    Full 3-level query understanding pipeline.

    Level 1 — Intent classification (LLM classifier)
      Prevents "Garbage In, Garbage Out" by routing before retrieval.
      RETRIEVAL → normal flow
      MEMORY    → skip Qdrant, answer from history
      HYBRID    → use history context + retrieve
      CHITCHAT  → skip Qdrant, respond directly

    Level 2 — Query understanding
      Entity resolution  → resolves pronouns/references
      Sub-query decomp   → breaks complex queries into parts

    Level 3 — Structured state injection
      Injects episodic summary into hybrid queries
      Tracks intent for downstream nodes

    All LLM calls use Haiku — fast and cheap.
    Total overhead: ~200-400ms, ~$0.0003 per query.
    """
    query        = state.get("query", "")
    chat_history = state.get("chat_history", [])
    episodic     = state.get("episodic_summary", "")
    topics       = state.get("topics_discussed", [])
    papers       = state.get("papers_referenced", [])

    llm        = get_llm(max_tokens=300)
    raw_client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    # ── Level 1: Intent classification ───────────────────
    intent = _classify_intent(query, chat_history, raw_client)
    print(f"[query_rewriter] intent     : {intent}")

    # ── MEMORY — skip retrieval entirely ──────────────────
    if intent == "MEMORY":
        print(f"[query_rewriter] → memory_generator (no Qdrant)")
        return {
            "intent"         : "MEMORY",
            "rewritten_query": query,
            "hyde_doc"       : "",
            "query_type"     : "memory",
        }

    # ── CHITCHAT — skip retrieval entirely ────────────────
    if intent == "CHITCHAT":
        print(f"[query_rewriter] → memory_generator (chitchat)")
        return {
            "intent"         : "CHITCHAT",
            "rewritten_query": query,
            "hyde_doc"       : "",
            "query_type"     : "chitchat",
        }

    # ── Level 2: Entity resolution ────────────────────────
    resolved_query = _extract_entities(query, chat_history, llm)
    if resolved_query != query:
        print(f"[query_rewriter] resolved   : '{resolved_query}'")

    # ── Level 2: Sub-query decomposition ─────────────────
    sub_queries     = _decompose_query(resolved_query, llm)
    retrieval_query = sub_queries[0]   # primary query for retrieval

    # ── Level 3: Inject episodic summary for HYBRID ───────
    # gives retrieval the benefit of conversation context
    if episodic and intent == "HYBRID":
        retrieval_query = (
            f"{retrieval_query} "
            f"[Context: {episodic[:200]}]"
        )
        print(f"[query_rewriter] hybrid context injected")

    # ── HyDE — only for RETRIEVAL and HYBRID ─────────────
    try:
        hyde_doc = _generate_hyde_document(retrieval_query, llm)

        if hyde_doc and len(hyde_doc) > 20:
            print(f"[query_rewriter] original   : '{query}'")
            print(f"[query_rewriter] HyDE doc   : '{hyde_doc[:100]}...'")
            return {
                "intent"         : intent,
                "rewritten_query": hyde_doc,
                "hyde_doc"       : hyde_doc,
                "query_type"     : "text",
            }

    except Exception as e:
        print(f"[query_rewriter] HyDE failed: {e} → falling back")

    # ── Fallback — HyDE failed, use resolved query ────────
    print(f"[query_rewriter] rewritten  : '{retrieval_query}'")
    return {
        "intent"         : intent,
        "rewritten_query": retrieval_query,
        "hyde_doc"       : "",
        "query_type"     : "text",
    }


# ============================================================
# NODE 1 — query_router
# ============================================================
def query_router(state: RAGState) -> dict:
    query      = state.get("query", "")
    image_path = state.get("image_path")

    raw_client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    if image_path:
        # user uploaded image — always visual
        query_type = "image"
    else:
        # LLM decides — no keywords, no hardcoding
        modality   = _classify_query_modality(query, raw_client)
        query_type = "image" if modality == "VISUAL" else "text"

    print(f"[router] modality={query_type} | query='{query[:80]}'")

    return {"query_type": query_type}


# ============================================================
# NODE 2A — text_retriever
# ============================================================
def text_retriever(state: RAGState) -> dict:
    """
    Embeds query with BGE → searches text_vector in Qdrant.
    Returns top 20 results (text chunks + figure captions).

    MEMORY CHANGE: uses rewritten_query not raw query.
    rewritten_query is self-contained for follow-up questions.
    For first questions rewritten_query == raw query.

    Both text chunks AND figure records return here because
    figure captions are embedded in text_vector slot.
    """
    # use rewritten_query for accurate follow-up retrieval
    query = state.get("rewritten_query") or state.get("query", "")
    if not query:
        return {"text_results": []}

    print(f"[text_retriever] embedding: '{query[:60]}'")
    query_vec = embed_text(query).tolist()

    print(f"[text_retriever] searching Qdrant text_vector...")
    results = search_by_text(query_vec, limit=20)

    print(f"[text_retriever] found {len(results)} results")
    return {"text_results": results}


# ============================================================
# NODE 2B — image_retriever
# ============================================================
def image_retriever(state: RAGState) -> dict:
    """
    Embeds uploaded image with CLIP → searches image_vector in Qdrant.
    Returns visually similar figures above MIN_IMAGE_SCORE (0.60).

    No memory changes needed — image search uses pixels not text.
    Chat history doesn't affect visual similarity.

    If no image path or file doesn't exist → returns empty list.
    Pipeline continues gracefully without crashing.
    """
    image_path = state.get("image_path")
    if not image_path or not Path(image_path).exists():
        print(f"[image_retriever] no valid image path")
        return {"image_results": []}

    print(f"[image_retriever] embedding image...")
    image_vec = embed_image(image_path).tolist()

    print(f"[image_retriever] searching Qdrant image_vector...")
    results = search_by_image(image_vec, limit=20)

    print(f"[image_retriever] found {len(results)} results above threshold")
    return {"image_results": results}


# ============================================================
# NODE 3 — reranker
# ============================================================
def reranker(state: RAGState) -> dict:
    """
    Merges, deduplicates, and reranks retrieval results.

    Step 1: merge text_results + image_results
    Step 2: deduplicate by (arxiv_id + section + page + figure_num)
    Step 3: cross-encoder reranking (primary) or bi-encoder (fallback)
    Step 4: return top 5

    MEMORY CHANGE: uses rewritten_query for scoring.
    Cross-encoder compares rewritten query against each chunk —
    more accurate for follow-up questions.

    Cross-encoder vs bi-encoder:
      Bi-encoder:    embed query separately, embed doc separately, compare
      Cross-encoder: reads query + doc TOGETHER → more accurate
      Cross-encoder only runs on top 20 → ~200ms, acceptable latency
    """
    text_results  = state.get("text_results", [])
    image_results = state.get("image_results", [])

    # use rewritten_query for accurate reranking
    query = state.get("rewritten_query") or state.get("query", "")

    # merge all candidates
    all_results = text_results + image_results

    if not all_results:
        print(f"[reranker] no results to rank")
        return {"ranked_results": []}

    # ── Step 1: deduplicate ────────────────────────────────
    # Dedup key: paper + section + page + figure_num
    # Same chunk appearing in both text and image results
    # gets counted only once
    seen   = set()
    unique = []

    for r in all_results:
        p   = r["payload"]
        key = (
            p.get("arxiv_id",   ""),
            p.get("section",    ""),
            p.get("page",        0),
            p.get("figure_num", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"[reranker] {len(all_results)} → {len(unique)} after dedup")

    # ── Step 2: rerank ─────────────────────────────────────
    if query and len(unique) > 1:
        try:
            from sentence_transformers import CrossEncoder

            # cross-encoder loads once then cached
            # ~80MB model, downloads on first run
            cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512
            )

            # build (query, document) pairs
            # truncate doc to 400 chars — fits in cross-encoder window
            pairs = [
                (query, r["payload"].get("text", "")[:400])
                for r in unique
            ]

            # single forward pass per pair
            # reads query + document TOGETHER → accurate relevance score
            ce_scores = cross_encoder.predict(pairs)

            # normalize to 0-1 range
            ce_min   = float(min(ce_scores))
            ce_max   = float(max(ce_scores))
            ce_range = ce_max - ce_min if ce_max != ce_min else 1.0

            for i, r in enumerate(unique):
                ce_normalized  = (float(ce_scores[i]) - ce_min) / ce_range
                # 30% original Qdrant score + 70% cross-encoder
                # heavy weight on cross-encoder = more precise ranking
                r["final_score"] = (0.3 * r["score"] +
                                    0.7 * ce_normalized)

            unique.sort(key=lambda x: x["final_score"], reverse=True)
            print(f"[reranker] cross-encoder done")

        except Exception as e:
            # graceful fallback to bi-encoder
            # pipeline never breaks even if cross-encoder fails to load
            print(f"[reranker] cross-encoder failed: {e}")
            print(f"[reranker] falling back to bi-encoder")
            query_vec   = embed_text(query)
            result_vecs = embed_texts_batch(
                [r["payload"].get("text", "") for r in unique]
            )
            scores = result_vecs @ query_vec
            for i, r in enumerate(unique):
                r["final_score"] = (0.4 * r["score"] +
                                    0.6 * float(scores[i]))
            unique.sort(key=lambda x: x["final_score"], reverse=True)
    else:
        for r in unique:
            r["final_score"] = r["score"]

    # top 5
    top5 = unique[:5]

    print(f"[reranker] top 5:")
    for i, r in enumerate(top5):
        p = r["payload"]
        print(f"  [{i+1}] {r['final_score']:.4f} | "
              f"{p['paper_title'][:40]} | {p['section'][:30]}")

    return {"ranked_results": [r for r in top5]}


# ============================================================
# NODE 4 — context_builder
# ============================================================
def context_builder(state: RAGState) -> dict:
    """
    Formats ranked results into context string for LLM.
    Also builds figures list for UI display.

    No memory changes needed here.
    context_builder only cares about ranked_results.
    Chat history is handled by generator.

    Three jobs:
      1. Format text chunks into numbered context string
      2. Collect figure records for UI display
      3. Fetch extra figures from top papers if < 3 figures found
    """
    ranked = state.get("ranked_results", [])

    if not ranked:
        return {
            "context": "No relevant information found.",
            "figures": [],
            "sources": [],
        }

    context_parts = []
    figures       = []
    sources       = []
    seen_papers   = set()

    for i, result in enumerate(ranked):
        p = result["payload"]

        # build numbered context block
        # format: [1] Paper: X
        #             Section: Y
        #             text content...
        context_parts.append(
            f"[{i+1}] Paper: {p['paper_title']}\n"
            f"    Section: {p['section']}\n"
            f"    {p['text'][:500]}"
        )

        # track unique papers for source citations
        if p["arxiv_id"] not in seen_papers:
            seen_papers.add(p["arxiv_id"])
            sources.append({
                "arxiv_id"   : p["arxiv_id"],
                "paper_title": p["paper_title"],
                "category"   : p["category"],
                "url"        : f"https://arxiv.org/abs/{p['arxiv_id']}",
            })

        # collect figures directly from ranked results
        if p["record_type"] == "figure" and p.get("image_path"):
            figures.append({
                "image_path" : p["image_path"],
                "caption"    : p["caption"],
                "paper_title": p["paper_title"],
                "arxiv_id"   : p["arxiv_id"],
                "score"      : result["final_score"],
            })

    # fetch extra figures if ranked results are mostly text chunks
    # ensures UI always has visuals to show alongside text answers
    if len(figures) < 3:
        for arxiv_id in list(seen_papers)[:2]:
            extra_figs = get_paper_figures(arxiv_id, limit=2)
            for f in extra_figs:
                p = f["payload"]
                if p.get("image_path") and len(figures) < 5:
                    figures.append({
                        "image_path" : p["image_path"],
                        "caption"    : p["caption"],
                        "paper_title": p["paper_title"],
                        "arxiv_id"   : p["arxiv_id"],
                        "score"      : 0.0,  # fetched not retrieved
                    })

    context = "\n\n".join(context_parts)

    print(f"[context_builder] context: {len(context)} chars")
    print(f"[context_builder] figures: {len(figures)}")
    print(f"[context_builder] sources: {len(sources)}")

    return {
        "context": context,
        "figures": figures,
        "sources": sources,
    }


# ============================================================
# NODE 5 — generator
# ============================================================


def is_context_relevant(context: str) -> bool:
    """
    Quick check — if context is empty or too short,
    retrieval found nothing useful for this query.
    """
    if not context:
        return False
    if context == "No relevant information found.":
        return False
    if len(context.strip()) < 100:
        return False
    return True


def generator(state: RAGState) -> dict:
    """
    Generates the final grounded answer using Claude.

    MEMORY CHANGES:
      - reads chat_history from state
      - builds messages with full conversation history
      - appends new turn to chat_history
      - returns updated chat_history

    LEVEL 3 CHANGES:
      - extracts topics discussed from this turn
      - tracks papers referenced
      - returns structured conversation state updates

    GROUNDING CHANGES:
      - stricter system prompt — no general knowledge allowed
      - context quality check before LLM call
      - explicit I don't know response for off-topic queries

    Why this makes it RAG not just a chatbot:
      System prompt strictly says "use ONLY provided context"
      LLM cannot use its training knowledge to answer
      Every claim must come from retrieved chunks
      This is what faithfulness measures in RAGAS

    Message structure sent to Claude:
      SystemMessage  ← instructions and rules
      HumanMessage   ← turn 1 user question (from history)
      AIMessage      ← turn 1 assistant answer (from history)
      ...
      HumanMessage   ← current question + retrieved context
    """
    query        = state.get("query", "")
    context      = state.get("context", "")
    chat_history = state.get("chat_history", [])

    # ── Guard: no query ───────────────────────────────────
    if not query:
        return {
            "answer"           : "Please provide a query.",
            "chat_history"     : chat_history,
            "topics_discussed" : state.get("topics_discussed", []),
            "papers_referenced": state.get("papers_referenced", []),
        }

    # ── Guard: context not relevant ───────────────────────
    # If retrieval returned nothing useful, don't call LLM
    # This prevents Claude from answering from general knowledge
    if not is_context_relevant(context):
        no_info_answer = (
            "I don't have information about this in my research "
            "papers database. My knowledge is limited to the AI "
            "research papers I have been given access to."
        )
        updated_history = chat_history + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": no_info_answer},
        ]
        return {
            "answer"           : no_info_answer,
            "chat_history"     : updated_history,
            "topics_discussed" : state.get("topics_discussed", []),
            "papers_referenced": state.get("papers_referenced", []),
        }

    # ── Stricter system prompt ────────────────────────────
    # Explicitly forbids using general training knowledge
    # Forces "I don't know" when context is insufficient
    system_prompt = """You are a research assistant with access ONLY to a specific database of AI research papers.

Answer questions using ONLY the provided context chunks from the database.

Rules:
- Cite the paper for every specific claim: "According to [Paper Name]..."
- If the retrieved context does not contain enough information to answer → say exactly:
  "I don't have information about this in my research papers database."
- NEVER use your general training knowledge to answer — only use the provided context
- If context is partially relevant, answer what you can and clearly flag what is missing
- Be technical and precise — the user is an AI/ML engineer
- Do not speculate or infer beyond what the context explicitly states
- If the user refers to something from previous conversation, use that context"""

    # ── Build message list with history ───────────────────
    # This is what gives the pipeline conversational memory.
    # Claude sees the full conversation, not just current question.
    messages = [SystemMessage(content=system_prompt)]

    # add previous conversation turns
    for turn in chat_history[-10:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    # add current question with retrieved context
    # context goes with current question — not in system prompt
    # this ensures grounding happens at question level
    messages.append(HumanMessage(content=(
        f"Context from research papers:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer based on the above context only:"
    )))

    print(f"[generator] calling LLM...")
    print(f"[generator] messages: {len(messages)} "
          f"({len(chat_history)} history turns)")

    llm      = get_llm(max_tokens=1500)
    response = llm.invoke(messages)
    answer   = response.content

    print(f"[generator] answer: {len(answer)} chars")

    # ── Build updated history ONCE with trim ──────────────
    updated_history = chat_history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]
    if len(updated_history) > 10:
        updated_history = updated_history[-10:]

    # ── Level 3: Extract topics + papers ──────────────────
    topic_extraction_prompt = (
        f"Extract 2-3 key technical topics from this Q&A as a "
        f"JSON array of short strings (3-5 words each).\n"
        f"Q: {query}\nA: {answer[:300]}\n\n"
        f"Topics (JSON array only):"
    )
    paper_ids = [s["arxiv_id"] for s in state.get("sources", [])]

    new_topics = []
    try:
        raw_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        topic_resp = raw_client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 80,
            messages   = [{"role": "user", "content": topic_extraction_prompt}]
        )
        content = topic_resp.content[0].text.strip()

        # bulletproof JSON parsing
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        new_topics = json.loads(content)
        if not isinstance(new_topics, list):
            new_topics = []

    except json.JSONDecodeError as e:
        print(f"[generator] topic JSON parse failed: {e}")
    except Exception as e:
        print(f"[generator] topic extraction failed: {e}")

    # ── Merge with existing structured state ──────────────
    existing_topics = state.get("topics_discussed", [])
    existing_papers = state.get("papers_referenced", [])

    all_topics = existing_topics + [
        t for t in new_topics if t not in existing_topics
    ]
    all_papers = existing_papers + [
        p for p in paper_ids if p not in existing_papers
    ]

    print(f"[generator] topics tracked : {all_topics}")
    print(f"[generator] papers tracked : {all_papers}")

    return {
        "answer"           : answer,
        "chat_history"     : updated_history,   # single build, trimmed
        "topics_discussed" : all_topics,
        "papers_referenced": all_papers,
    }

# ============================================================
# NODE 5 — memory Generator (for MEMORY and CHITCHAT intents)
# ============================================================

def memory_generator(state: RAGState) -> dict:
    """
    Handles MEMORY and CHITCHAT intents.
    MEMORY   → answers from chat history + structured state
    CHITCHAT → politely declines, stays in scope
    """
    query        = state.get("query", "")
    chat_history = state.get("chat_history", [])
    intent       = state.get("intent", "MEMORY")
    episodic     = state.get("episodic_summary", "")
    topics       = state.get("topics_discussed", [])
    papers       = state.get("papers_referenced", [])

    # ── CHITCHAT — decline politely, don't call LLM ───────
    if intent == "CHITCHAT":
        answer = (
            "I'm a research assistant specialized in AI research papers. "
            "I can only answer questions about the papers in my database. "
            "Try asking me about transformers, LLMs, RAG, diffusion models, "
            "or any other AI research topic!"
        )
        updated_history = chat_history + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": answer},
        ]
        print(f"[memory_generator] chitchat declined politely")
        return {
            "answer"           : answer,
            "chat_history"     : updated_history,
            "context"          : "",
            "figures"          : [],
            "sources"          : [],
            "topics_discussed" : state.get("topics_discussed", []),
            "papers_referenced": state.get("papers_referenced", []),
        }

    # ── MEMORY — answer from conversation history ─────────
    llm = get_llm(max_tokens=800)

    # build context from structured state
    structured_context = ""
    if episodic:
        structured_context += f"Conversation summary: {episodic}\n\n"
    if topics:
        structured_context += f"Topics we discussed: {', '.join(topics)}\n"
    if papers:
        structured_context += f"Papers we referenced: {', '.join(papers)}\n"

    system = (
        "You are a research assistant. Answer based ONLY on the "
        "conversation history and summary provided. "
        "Do not retrieve or invent new information. "
        "If the conversation history does not contain enough "
        "information to answer, say so clearly."
    )

    messages = [SystemMessage(content=system)]

    # inject structured context
    if structured_context:
        messages.append(HumanMessage(content=(
            f"[Conversation context]\n{structured_context}"
        )))
        messages.append(AIMessage(content=(
            "Understood. I have the context of our previous discussion."
        )))

    # add recent turns
    for turn in chat_history[-10:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)
    answer   = response.content

    updated_history = chat_history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]

    print(f"[memory_generator] answered from memory | intent={intent}")

    return {
        "answer"           : answer,
        "chat_history"     : updated_history,
        "context"          : "",
        "figures"          : [],
        "sources"          : [],
        "topics_discussed" : state.get("topics_discussed", []),
        "papers_referenced": state.get("papers_referenced", []),
    }



# ============================================================
# NODE 6 — query classifier
# ============================================================

def _classify_query_modality(
    query  : str,
    client : anthropic.Anthropic,
) -> str:
    """
    LLM-based modality classifier.
    Decides whether query needs visual content or text only.
    
    TEXT   → explanation, comparison, definition, how-it-works
    VISUAL → explicitly wants figures, diagrams, architecture visuals
    
    Cost: ~$0.00001 per call (10 tokens max)
    """
    response = client.messages.create(
        model      = "claude-haiku-4-5-20251001",
        max_tokens = 10,
        messages   = [{
            "role"   : "user",
            "content": (
                f"Does this query need visual content (figures, diagrams, "
                f"images) or just text explanation?\n\n"
                f"Reply with only one word: TEXT or VISUAL\n\n"
                f"Examples:\n"
                f"'how does attention work?' → TEXT\n"
                f"'show me the transformer architecture diagram' → VISUAL\n"
                f"'explain CLIP' → TEXT\n"
                f"'what does the Mamba architecture look like?' → VISUAL\n"
                f"'compare GPT-3 vs LLaMA' → TEXT\n"
                f"'draw the attention mechanism' → VISUAL\n\n"
                f"Query: {query}\n\n"
                f"Answer:"
            )
        }]
    )

    modality = response.content[0].text.strip().upper()
    return modality if modality in ("TEXT", "VISUAL") else "TEXT"
