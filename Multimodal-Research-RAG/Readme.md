# 🔬 Multimodal RAG Pipeline over AI Research Papers

A production-grade Retrieval-Augmented Generation system built over **21 AI research papers** (sample dataset) with multimodal support, conversational memory, and a 3-level query understanding pipeline.

> **Note:** This project currently runs on a sample dataset of 21 papers across 7 categories. The full pipeline is designed to scale to 200+ papers, expansion is planned once all PDFs are verified for correct downloads and extraction quality.

---

## 🎯 What it does

Ask natural language questions about AI research papers and get grounded, cited answers with relevant figures — like having a research assistant that has read every paper for you.

```
"How does attention work in the Transformer?"
→ Retrieves relevant chunks from Attention Is All You Need
→ Reranks with cross-encoder
→ Generates grounded answer with citations
→ Shows Figure 1 (Transformer architecture) inline
```

---

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│         3-Level Query Understanding          │
│                                              │
│  Level 1: Intent Classification (LLM)        │
│    RETRIEVAL → search papers                 │
│    MEMORY    → answer from history only      │
│    HYBRID    → history context + search      │
│    CHITCHAT  → polite decline                │
│                                              │
│  Level 2: Entity Resolution                  │
│    (RETRIEVAL + HYBRID only)                 │
│    "how does it work?" →                     │
│    "how does scaled dot-product              │
│     attention work?"                         │
│                                              │
│  Level 2: Sub-query Decomposition            │
│    (RETRIEVAL + HYBRID only)                 │
│    "compare Mamba vs Transformer" →          │
│    ["Mamba architecture",                    │
│     "Transformer architecture",              │
│     "comparison of both"]                   │
│                                              │
│  HyDE: Hypothetical Document Embedding       │
│    (RETRIEVAL + HYBRID only)                 │
│    NOT applied to MEMORY or CHITCHAT         │
└─────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────┐
│              Intent-based Routing                     │
│                                                       │
│  MEMORY / CHITCHAT → memory_generator (no Qdrant)    │
│  RETRIEVAL / HYBRID → query_router                   │
│                           ↓                          │
│                  LLM Modality Classifier              │
│                  TEXT   → text_retriever only        │
│                  VISUAL → text_retriever              │
│                         + image_retriever            │
└──────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│         LangGraph Pipeline (8 nodes)         │
│                                              │
│  query_rewriter                              │
│      ↓ (MEMORY/CHITCHAT)                    │
│  memory_generator → END                     │
│      ↓ (RETRIEVAL/HYBRID)                   │
│  query_router →                              │
│  text_retriever / image_retriever →          │
│  reranker → context_builder → generator     │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│         Two-Tier Memory System               │
│                                              │
│  Tier 1: Recent turns (verbatim)             │
│  Tier 2: Episodic summary (compressed)       │
│  Summarization every 10 turns                │
└─────────────────────────────────────────────┘
    ↓
Grounded Answer + Citations + Figures
```

---

## 📊 Dataset (Sample)

| Stat | Value |
|------|-------|
| Total papers (current) | 21 PDFs |
| Categories | 7 |
| Total vectors | 1,224 |

> **Planned:** Scale to 239 verified PDFs once all downloads and extractions are validated.

**Categories:**
- 🤖 Transformers & Attention
- 📈 LLMs & Scaling
- 🔍 RAG & Retrieval
- ⚡ Efficient & Fine-tuning
- 🎯 RL & Alignment
- 🖼️ CV & Multimodal
- 🎨 Diffusion & Generation

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Extraction | Marker (LaTeX preservation) |
| Text Embeddings | BGE-base-en-v1.5 (768-dim) |
| Image Embeddings | CLIP ViT-B/32 (512-dim) |
| Vector Database | Qdrant (local) |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| LLM | Claude Haiku (Anthropic) |
| Pipeline | LangGraph |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## 🔄 Pipeline Details

### Query Understanding (3 Levels)

**Level 1 — Intent Classification**

LLM-based classifier routes each query before any retrieval:

| Intent | Action |
|--------|--------|
| `RETRIEVAL` | Entity resolution → decomposition → HyDE → Qdrant search |
| `MEMORY` | Skip Qdrant entirely → answer from conversation history |
| `HYBRID` | Entity resolution → HyDE → Qdrant search + history context |
| `CHITCHAT` | Skip Qdrant entirely → polite decline |

**Level 2 — Entity Resolution** *(RETRIEVAL + HYBRID only)*

Resolves pronouns and references using conversation history:
```
"how does it work?" + [context: transformer]
→ "how does the transformer architecture work?"
```

**Level 2 — Sub-query Decomposition** *(RETRIEVAL + HYBRID only)*

Breaks complex comparison queries into targeted searches:
```
"compare attention in transformers vs mamba"
→ ["attention mechanism transformers",
   "attention mechanism mamba",
   "comparison transformer mamba architectures"]
```

**HyDE — Hypothetical Document Embedding** *(RETRIEVAL + HYBRID only)*

Generates a hypothetical answer passage and embeds that instead of the raw query. Vector search works better comparing answer-to-answer than question-to-answer.

**HyDE is NOT applied to MEMORY or CHITCHAT** — these intents skip retrieval entirely.

---

### Query Router — Modality Classification

After intent routing, the `query_router` node uses an LLM to understand the question and decide whether it needs text or visual content:

| Modality | When | Retriever |
|----------|------|-----------|
| `TEXT` | Explanation, definition, comparison | `text_retriever` only |
| `VISUAL` | Explicitly wants figures, diagrams, architecture visuals | `text_retriever` + `image_retriever` |

```
"how does attention work?"                    → TEXT   → text_retriever only
"show me the transformer architecture"        → VISUAL → both retrievers
"explain CLIP training"                       → TEXT   → text_retriever only
"what does the Mamba architecture look like?" → VISUAL → both retrievers
```

No keyword matching — the LLM understands intent naturally.

---

### Memory System

```
Turn 1-10  : recent_turns stored verbatim
Turn 10    : should_summarize() → True
             old turns compressed into episodic_summary
             recent turns kept verbatim
Turn 11-20 : continues accumulating
Turn 20    : summarize() folds into existing summary
```

Benefits:
- Never hits context window limit
- Preserves recent turns for exact recall
- Compresses older turns to save tokens
- Structured state tracks topics + papers referenced across turns

---

### Retrieval Strategy

- Text queries → BGE `text_vector` search only
- Visual queries → CLIP `image_vector` + BGE `text_vector`
- Cross-encoder reranker scores top-20 → returns top-5
- Figures served as supplementary — never compete with text chunks for top-5 slots

---

## 📈 Evaluation (RAGAS)

> **Note:** Evaluation was run on the sample dataset of 21 papers. Results below reflect the pipeline before the final fixes (LLM modality classifier, figure retrieval fix) were applied. Re-evaluation is planned once Anthropic API credits are added — scores are expected to improve significantly especially for Answer Relevancy and Context Precision.

Evaluated on 21 questions across all 7 categories:

| Metric | Score | Notes |
|--------|-------|-------|
| Faithfulness | 0.75 | Answers grounded in retrieved context |
| Answer Relevancy | 0.31 | Affected by figure-heavy retrieval (now fixed) |
| Context Precision | 0.34 | Affected by figure-heavy retrieval (now fixed) |
| Context Recall | 0.40 | Small sample dataset |

**Best performing category — Transformers & Attention:**

| Metric | Score |
|--------|-------|
| Faithfulness | 0.90 |
| Answer Relevancy | 0.89 |
| Context Precision | 0.75 |
| Context Recall | 0.70 |

Transformers & Attention scores are representative of pipeline performance on text-rich papers. Lower scores in other categories are caused by figure-heavy papers with fewer text chunks — this is now fixed by the LLM modality classifier routing text queries away from figure retrieval.

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Anthropic API key

### Installation

```bash
# clone repo
git clone https://github.com/yourusername/multimodal-rag
cd multimodal-rag

# create virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

### Run

```bash
# interactive terminal mode
python -m rag_pipeline.run

# single query
python -m rag_pipeline.run --query "How does attention work?"

# Streamlit UI
streamlit run app.py
```

### Evaluate

```bash
# full evaluation
python evaluate.py

# re-run RAGAS only using saved pipeline outputs
python evaluate.py --skip-pipeline

# check which papers are in collection
python evaluate.py --check-only
```

---

## 📁 Project Structure

```
pathology_rag_pipeline/
├── rag_pipeline/
│   ├── state.py              ← RAGState TypedDict
│   ├── nodes.py              ← all 8 nodes + ConversationMemory
│   ├── graph.py              ← LangGraph assembly + intent routing
│   ├── models.py             ← CLIP + BGE model loaders
│   ├── retriever.py          ← Qdrant client + search
│   └── run.py                ← entry point + interactive loop
├── Notebooks/
│   └── Extraction/           ← Marker extraction notebooks (Kaggle)
├── embedding_source_data/    ← embedding pipeline scripts
├── eval_output/              ← RAGAS evaluation results
├── marker_output/            ← Marker extraction output
├── qdrant_backup/            ← local Qdrant storage
├── Sample_Data/              ← sample PDFs + download log
├── Source_Data/              ← full dataset PDFs
├── Documentation/            ← project notes
├── app.py                    ← Streamlit chat UI
├── evaluate.py               ← RAGAS evaluation script
├── add_papers.py             ← add new papers to collection
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🗺️ Roadmap

- [ ] Verify and fix all 239 PDF downloads (check truncated arxiv IDs)
- [ ] Re-extract full dataset with Marker on Kaggle GPU
- [ ] Re-embed full 239 papers into Qdrant
- [ ] Re-run RAGAS evaluation with updated pipeline fixes
- [ ] Deploy Streamlit UI to Hugging Face Spaces
- [ ] Add FastAPI REST endpoint
- [ ] Add Qdrant Cloud support

---

## 💡 Production Considerations

| Current | Production Equivalent |
|---------|----------------------|
| Local Qdrant | Qdrant Cloud / Pinecone |
| Manual `add_papers.py` | Airflow DAG + S3 trigger |
| Streamlit UI | React / Next.js frontend |
| `run.py` interactive | FastAPI REST endpoint |
| Local file storage | S3 / GCS bucket |

---

## 📝 License

MIT License — free to use, modify, and distribute.
