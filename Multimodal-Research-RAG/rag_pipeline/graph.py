from langgraph.graph import StateGraph, START, END
from .state import RAGState
from .nodes import (
    query_rewriter,
    query_router,
    text_retriever,
    image_retriever,
    reranker,
    context_builder,
    generator,
    memory_generator,    # NEW
)


def route_after_rewriter(state: RAGState) -> str:
    """
    Level 1 routing — based on intent from query_rewriter.
    MEMORY / CHITCHAT → memory_generator (skip Qdrant entirely)
    everything else   → query_router (normal retrieval flow)
    """
    intent = state.get("intent", "RETRIEVAL")
    if intent in ("MEMORY", "CHITCHAT"):
        return "memory_generator"
    return "query_router"


def route_query(state: RAGState) -> list[str]:
    """Route to text/image/both retrievers."""
    query_type = state.get("query_type", "text")
    has_image  = bool(state.get("image_path"))

    if query_type == "image" and has_image:
        if state.get("query", "").strip():
            return ["text_retriever", "image_retriever"]
        return ["image_retriever"]
    return ["text_retriever"]


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    # ── nodes ──────────────────────────────────────────────
    graph.add_node("query_rewriter",   query_rewriter)
    graph.add_node("query_router",     query_router)
    graph.add_node("text_retriever",   text_retriever)
    graph.add_node("image_retriever",  image_retriever)
    graph.add_node("reranker",         reranker)
    graph.add_node("context_builder",  context_builder)
    graph.add_node("generator",        generator)
    graph.add_node("memory_generator", memory_generator)   # NEW

    # ── edges ──────────────────────────────────────────────
    graph.add_edge(START, "query_rewriter")

    # intent-based routing after rewriter
    graph.add_conditional_edges(
        "query_rewriter",
        route_after_rewriter,
        {
            "memory_generator": "memory_generator",
            "query_router"    : "query_router",
        }
    )

    # retriever routing
    graph.add_conditional_edges(
        "query_router",
        route_query,
        {
            "text_retriever" : "text_retriever",
            "image_retriever": "image_retriever",
        }
    )

    graph.add_edge("text_retriever",  "reranker")
    graph.add_edge("image_retriever", "reranker")
    graph.add_edge("reranker",        "context_builder")
    graph.add_edge("context_builder", "generator")
    graph.add_edge("generator",       END)
    graph.add_edge("memory_generator", END)    # NEW

    compiled = graph.compile()

    print("[+] LangGraph pipeline compiled")
    print("    Nodes: query_rewriter → [memory_generator | query_router]")
    print("           → retriever(s) → reranker → context_builder → generator")

    return compiled


rag_graph = build_graph()