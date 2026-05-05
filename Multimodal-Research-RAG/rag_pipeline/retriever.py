"""
retriever.py — Qdrant search functions

Handles all communication with Qdrant.
Single source of truth for search logic.
"""

from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ── Config ─────────────────────────────────────────────────
QDRANT_PATH     = Path("qdrant_backup/qdrant_storage")   # local restored path
COLLECTION_NAME = "arxiv_rag"
TOP_K_RETRIEVAL = 50    # increased from 20 — HyDE needs wider net
MIN_IMAGE_SCORE = 0.60  # threshold below which image match is unreliable

# ── Singleton client ───────────────────────────────────────
_client = None


def get_client() -> QdrantClient:
    """Get or create Qdrant client. Singleton."""
    global _client
    if _client is None:
        if not QDRANT_PATH.exists():
            raise FileNotFoundError(
                f"Qdrant storage not found at '{QDRANT_PATH}'. "
                f"Unzip qdrant_backup.zip first:\n"
                f"  Expand-Archive qdrant_backup.zip ."
            )
        _client = QdrantClient(path=str(QDRANT_PATH))
        info = _client.get_collection(COLLECTION_NAME)
        print(f"[+] Qdrant connected — {info.points_count:,} points")
    return _client


def search_by_text(query_vec: list, limit: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Search text_vector slot.
    Returns both text chunks AND figure records
    (figures have caption text embedded in text_vector).
    """
    client  = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="text_vector",
        limit=limit,
        with_payload=True,
    ).points

    return [
        {"score": r.score, "payload": r.payload}
        for r in results
    ]


def search_by_image(query_vec: list, limit: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Search image_vector slot.
    Only returns figure records — text records have zero
    image vectors and score ~0.0, so they naturally filter out.
    Applies MIN_IMAGE_SCORE threshold for quality control.
    """
    client  = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="image_vector",
        limit=limit,
        with_payload=True,
        score_threshold=MIN_IMAGE_SCORE,
    ).points

    return [
        {"score": r.score, "payload": r.payload}
        for r in results
        if r.payload.get("record_type") == "figure"
    ]


def get_paper_figures(arxiv_id: str, limit: int = 5) -> list[dict]:
    """
    Fetch figures from a specific paper by arxiv_id.
    Used by context_builder to attach figures to text answers.
    """
    client  = get_client()
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="arxiv_id",
                    match=MatchValue(value=arxiv_id)
                ),
                FieldCondition(
                    key="record_type",
                    match=MatchValue(value="figure")
                ),
            ]
        ),
        limit=limit,
        with_payload=True,
    )[0]

    return [
        {"score": 1.0, "payload": r.payload}
        for r in results
        if r.payload.get("caption")
        and not r.payload["caption"].startswith("Figure from")
    ]