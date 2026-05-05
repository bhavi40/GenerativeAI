"""
models.py — CLIP and BGE model loader

Singleton pattern — models load once and stay in memory.
Importing this module anywhere in the pipeline gives the
same model instances — no double loading.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Singleton instances ────────────────────────────────────
_clip_model     = None
_clip_processor = None
_bge_model      = None


def load_models():
    """Load CLIP and BGE models. Safe to call multiple times."""
    global _clip_model, _clip_processor, _bge_model

    if _clip_model is None:
        print(f"[+] Loading CLIP on {DEVICE}...")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print("    CLIP ready — 512-dim")

    if _bge_model is None:
        print(f"[+] Loading BGE on {DEVICE}...")
        _bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
        print("    BGE ready  — 768-dim")


def embed_image(image_path: str) -> np.ndarray:
    """
    Embed one image with CLIP. Returns (512,) float32 array.

    Resizes to 224x224 before loading — reduces memory,
    matches what CLIP processes internally anyway.
    """
    load_models()
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
    except Exception:
        img = Image.new("RGB", (224, 224), color=0)

    with torch.no_grad():
        inputs   = _clip_processor(images=[img], return_tensors="pt").to(DEVICE)
        features = _clip_model.get_image_features(**inputs)

        # handle newer transformers returning output object
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output

        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0].astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """
    Embed one text string with BGE. Returns (768,) float32 array.

    BGE prefix improves retrieval quality for passage search.
    """
    load_models()
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    vec = _bge_model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec[0].astype(np.float32)


def embed_texts_batch(texts: list[str]) -> np.ndarray:
    """Embed multiple texts. Returns (N, 768) float32 array."""
    load_models()
    prefixed = [
        f"Represent this sentence for searching relevant passages: {t}"
        for t in texts
    ]
    return _bge_model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
