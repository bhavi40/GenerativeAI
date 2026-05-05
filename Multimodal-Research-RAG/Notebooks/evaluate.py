import os
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set in .env")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from rag_pipeline.retriever import get_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from anthropic import Anthropic
import re
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper  
from ragas.embeddings import LangchainEmbeddingsWrapper



# ── Config ──────────────────────────────────────────────────
COLLECTION_NAME = "arxiv_rag"   # ← updated
OUTPUT_DIR      = Path("eval_output")
RESULTS_FILE    = OUTPUT_DIR / "eval_results.json"
REPORT_FILE     = OUTPUT_DIR / "eval_report.txt"
SLEEP_BETWEEN   = 5    # seconds between queries
# ────────────────────────────────────────────────────────────


# ============================================================
# SECTION 1 — Test Questions
# ============================================================

ALL_TEST_QUESTIONS = [

    # ── CV & Multimodal ───────────────────────────────────
    {
    "question" : "How does LLaVA-1.5 use MLP connectors to improve vision language alignment?",
    "category" : "CV & Multimodal",
    "arxiv_id" : "2310.03744",   # ← LLaVA-1.5 confirmed working
    "gt_source": "generate",
    },
    {
        "question" : "What improvements does LLaVA-1.5 make over the original LLaVA model?",
        "category" : "CV & Multimodal",
        "arxiv_id" : "2310.03744",
        "gt_source": "abstract",
    },
    {
        "question" : "How does CLIP learn visual representations from natural language supervision?",
        "category" : "CV & Multimodal",
        "arxiv_id" : "2103.0002",
        "gt_source": "abstract",
    },

    # ── Diffusion & Generation ────────────────────────────
    {
        "question" : "How does Stable Video Diffusion extend image diffusion models to video generation?",
        "category" : "Diffusion & Generation",
        "arxiv_id" : "2403.12015",
        "gt_source": "abstract",
    },
    {
        "question" : "How does DragGAN allow users to interactively manipulate images using point-based control?",
        "category" : "Diffusion & Generation",
        "arxiv_id" : "2305.16103",
        "gt_source": "abstract",
    },
    {
        "question" : "How does Imagic perform text-based editing on real images using diffusion models?",
        "category" : "Diffusion & Generation",
        "arxiv_id" : "2209.1443",
        "gt_source": "abstract",
    },

    # ── Efficient & Fine-tuning ───────────────────────────
    {
        "question" : "How does FlashAttention-2 improve parallelism and reduce memory compared to standard attention?",
        "category" : "Efficient & Fine-tuning",
        "arxiv_id" : "2307.08691",
        "gt_source": "abstract",
    },
    {
        "question" : "What surprising capabilities does GPT-4 demonstrate according to the Sparks of AGI paper?",
        "category" : "Efficient & Fine-tuning",
        "arxiv_id" : "2304.01196",
        "gt_source": "abstract",
    },
    {
    "question" : "What is the tiling technique in FlashAttention and how does it reduce memory IO?",
    "category" : "Efficient & Fine-tuning",
    "arxiv_id" : "2307.08691",
    "gt_source": "generate",
    },

    # ── LLMs & Scaling ────────────────────────────────────
    {
        "question" : "What is the training strategy and scaling approach used in DeepSeek LLM?",
        "category" : "LLMs & Scaling",
        "arxiv_id" : "2401.02954",
        "gt_source": "abstract",
    },
    {
        "question" : "How does Gemini 1.5 achieve multimodal understanding across millions of tokens?",
        "category" : "LLMs & Scaling",
        "arxiv_id" : "2403.0553",
        "gt_source": "abstract",
    },
    {
        "question" : "How does GPT-3 perform few-shot learning without any gradient updates?",
        "category" : "LLMs & Scaling",
        "arxiv_id" : "2005.14165",
        "gt_source": "abstract",
    },

    # ── RAG & Retrieval ───────────────────────────────────
    {
        "question" : "How does retrieval augmented generation combine parametric and non-parametric memory?",
        "category" : "RAG & Retrieval",
        "arxiv_id" : "2005.11401",
        "gt_source": "abstract",
    },
    {
        "question" : "How does FLARE decide when to retrieve and what to retrieve during generation?",
        "category" : "RAG & Retrieval",
        "arxiv_id" : "2305.14283",
        "gt_source": "abstract",
    },
    {
        "question" : "What metrics does RAGAS use to evaluate retrieval augmented generation pipelines?",
        "category" : "RAG & Retrieval",
        "arxiv_id" : "2309.01431",
        "gt_source": "abstract",
    },

    # ── RL & Alignment ────────────────────────────────────
    {
        "question" : "How does many-shot in-context learning differ from few-shot learning in language models?",
        "category" : "RL & Alignment",
        "arxiv_id" : "2404.03715",
        "gt_source": "abstract",
    },
    {
        "question" : "How does Voyager use LLMs to enable open-ended learning in Minecraft?",
        "category" : "RL & Alignment",
        "arxiv_id" : "2305.16291",
        "gt_source": "abstract",
    },
    {
        "question" : "How does ChatDev use communicative agents to automate software development?",
        "category" : "RL & Alignment",
        "arxiv_id" : "2307.09009",
        "gt_source": "abstract",
    },

    # ── Transformers & Attention ──────────────────────────
    {
        "question" : "How does the original transformer architecture use attention to replace recurrence?",
        "category" : "Transformers & Attention",
        "arxiv_id" : "1706.03762",
        "gt_source": "abstract",
    },
    {
        "question" : "How does Mamba achieve linear-time sequence modeling without attention?",
        "category" : "Transformers & Attention",
        "arxiv_id" : "2312.00752",
        "gt_source": "abstract",
    },
    {
        "question" : "How does the Reformer reduce the memory and computational cost of transformers?",
        "category" : "Transformers & Attention",
        "arxiv_id" : "2001.04451",
        "gt_source": "abstract",
    },
]

# ============================================================
# SECTION 2 — Pre-flight Check
# ============================================================
def check_available_papers(questions: list) -> list:
    from rag_pipeline.retriever import get_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_client()

    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"[+] Collection '{COLLECTION_NAME}': {info.points_count} points")
    except Exception as e:
        print(f"ERROR: Cannot connect to collection '{COLLECTION_NAME}': {e}")
        sys.exit(1)

    print(f"\n[+] Checking papers in collection...\n")

    all_ids       = list({q["arxiv_id"] for q in questions})
    available_ids = set()

    for arxiv_id in sorted(all_ids):
        try:
            results = client.query_points(
                collection_name = COLLECTION_NAME,
                query           = [0.0] * 768,
                using           = "text_vector",
                query_filter    = Filter(must=[
                    FieldCondition(
                        key  = "arxiv_id",
                        match= MatchValue(value=arxiv_id)
                    )
                ]),
                limit           = 1,
                with_payload    = False,
            ).points

            q     = next((q for q in questions if q["arxiv_id"] == arxiv_id), None)
            label = f"[{q['category']}]" if q else ""

            if results:
                available_ids.add(arxiv_id)
                print(f"  ✅  {arxiv_id}  {label}")
            else:
                print(f"  ❌  {arxiv_id}  {label}  — not in collection")

        except Exception as e:
            print(f"  ⚠️   {arxiv_id} — check failed: {e}")

    available = [q for q in questions if q["arxiv_id"] in available_ids]
    missing   = [q for q in questions if q["arxiv_id"] not in available_ids]

    print(f"\n  Available : {len(available)} questions")
    print(f"  Skipped   : {len(missing)} questions")

    if not available:
        print(f"\nERROR: No questions available.")
        print(f"       Collection '{COLLECTION_NAME}' may be empty or wrong name.")
        sys.exit(1)

    return available          # ← THIS was missing

# ============================================================
# SECTION 3 — Run Pipeline with Context Capture
# ============================================================

def run_query_with_contexts(query: str) -> dict:
    """
    Run the full pipeline and return BOTH the answer AND the
    exact chunks that were used to generate it.

    Includes all new state fields from the updated pipeline:
      - intent, hyde_doc, episodic_summary
      - topics_discussed, papers_referenced
      - rewritten_query

    No chat_history passed — evaluation queries are stateless.
    Each question is independent — no conversation context.
    """
    from rag_pipeline.graph import rag_graph

    initial_state = {
        # ── core fields ───────────────────────────────
        "query"            : query,
        "image_path"       : None,
        "query_type"       : "",
        "text_results"     : [],
        "image_results"    : [],
        "ranked_results"   : [],
        "context"          : "",
        "figures"          : [],
        "answer"           : "",
        "sources"          : [],
        # ── memory fields (empty for eval) ────────────
        "chat_history"     : [],
        "episodic_summary" : "",
        "topics_discussed" : [],
        "papers_referenced": [],
        # ── query understanding fields ─────────────────
        "intent"           : "",
        "rewritten_query"  : "",
        "hyde_doc"         : "",
    }

    final_state = rag_graph.invoke(initial_state)

    # ── extract exact contexts the LLM used ───────────
    ranked   = final_state.get("ranked_results", [])
    contexts = []
    for r in ranked:
        if isinstance(r, dict):
            text = (
                r.get("payload", {}).get("text", "")
                or r.get("text", "")
            )
            if text:
                contexts.append(text)

    # fallback: parse from context string if ranked_results empty
    raw_context = final_state.get("context", "")
    if not contexts and raw_context:
        import re
        parts = re.split(r'\[\d+\]', raw_context)
        for part in parts[1:]:
            lines = part.strip().split('\n')
            text_lines = [
                l.strip() for l in lines
                if l.strip()
                and not l.strip().startswith('Paper:')
                and not l.strip().startswith('Section:')
            ]
            if text_lines:
                contexts.append(' '.join(text_lines))

    return {
        "answer"  : final_state.get("answer", ""),
        "contexts": contexts,
        "sources" : final_state.get("sources", []),
        "intent"  : final_state.get("intent", ""),
    }


# ============================================================
# SECTION 4 — Ground Truth Generator
# ============================================================

def get_llm(max_tokens: int = 300):
    return ChatAnthropic(
        model  ="claude-haiku-4-5-20251001",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=max_tokens,
    )


def find_metadata_file() -> Path:
    """Auto-detect metadata.jsonl location."""
    candidates = [
        Path("marker_output/output/metadata.jsonl"),
        Path("data/metadata.jsonl"),
        Path("metadata.jsonl"),
        Path("Source_Data/Data/metadata.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            print(f"[+] metadata.jsonl found at: {path}")
            return path
    return None



def clean_ground_truth(text: str) -> str:
    """
    Strip author names, emails, affiliations that Marker 
    sometimes includes before the abstract text.
    """
    lines  = text.split('\n')
    clean  = []
    skip   = False

    for line in lines:
        # email line signals author block
        if re.search(r'[\w.\-]+@[\w.\-]+\.\w+', line):
            skip = True
            continue
        # short lines during skip = affiliation/institution
        if skip and len(line.strip()) < 60 and line.strip():
            continue
        # long line = real content, end of author block
        if skip and len(line.strip()) > 80:
            skip = False
        if not skip:
            clean.append(line)

    result = '\n'.join(clean).strip()
    # if cleaning removed too much — return original
    return result if len(result) > 100 else text

def load_abstract(arxiv_id: str) -> str:
    """
    Load abstract from metadata.jsonl.
    Handles merged section names like "Abstract / 1 Introduction".
    Auto-detects metadata file location.
    """
    metadata_file = find_metadata_file()
    if not metadata_file:
        print(f"    WARNING: metadata.jsonl not found in any known location")
        return ""

    best_match = ""
    with open(metadata_file, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("arxiv_id") != arxiv_id:
                continue
            if rec.get("record_type") != "text":
                continue

            section = rec.get("section", "").lower()

            # handles merged names: "Abstract / 1 Introduction"
            if "abstract" in section:
                text = rec.get("text", "")
                # prefer shorter (purer) abstract
                if not best_match or len(text) < len(best_match):
                    best_match = text

    return clean_ground_truth(best_match)


def generate_ground_truth(question: str, arxiv_id: str) -> str:
    """
    Auto-generate ground truth using Claude.
    Uses top 3 chunks from the specific paper.
    """
    from rag_pipeline.models import embed_text
    from rag_pipeline.retriever import get_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client    = get_client()
    query_vec = embed_text(question).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,    # ← uses config at top
        query          =query_vec,
        using          ="text_vector",
        query_filter   =Filter(must=[
            FieldCondition(
                key  ="arxiv_id",
                match=MatchValue(value=arxiv_id)
            )
        ]),
        limit          =3,
        with_payload   =True,
    ).points

    if not results:
        print(f"    WARNING: no chunks found for arxiv_id={arxiv_id}")
        return ""

    context = "\n\n".join([
        r.payload.get("text", "") for r in results
    ])

    llm      = get_llm(max_tokens=400)
    messages = [
        SystemMessage(content=(
            "You are a research assistant. Write a concise factual answer "
            "to the question using ONLY the provided context. "
            "3-5 sentences. No speculation. Be specific and technical."
        )),
        HumanMessage(content=(
            f"Context from paper {arxiv_id}:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def get_ground_truth(question_data: dict) -> str:
    """Get ground truth — from abstract or auto-generated."""
    gt_source = question_data["gt_source"]
    arxiv_id  = question_data["arxiv_id"]
    question  = question_data["question"]

    if gt_source == "abstract":
        gt = load_abstract(arxiv_id)
        if gt:
            return gt
        print(f"    Abstract not found for {arxiv_id}, generating...")

    return generate_ground_truth(question, arxiv_id)


# ============================================================
# SECTION 5 — Run Pipeline Queries
# ============================================================

def run_evaluation_queries(questions: list) -> list:
    """Run all questions through pipeline, collect eval data."""
    eval_data = []

    print(f"\n[+] Running {len(questions)} evaluation queries...")
    print(f"    Est. time: ~{len(questions) * 15 // 60 + 3} minutes\n")

    for i, q_data in enumerate(questions):
        question = q_data["question"]
        print(f"\n  [{i+1:02d}/{len(questions)}] {question[:65]}...")

        try:
            # run pipeline — get answer + exact contexts used
            result   = run_query_with_contexts(question)
            answer   = result["answer"]
            contexts = result["contexts"]
            intent   = result["intent"]

            print(f"    intent={intent} | "
                  f"answer={len(answer)}ch | "
                  f"contexts={len(contexts)}")

            if not contexts:
                print(f"    WARNING: no contexts — answer may be from memory/chitchat")

            # generate ground truth
            print(f"    generating ground truth...")
            ground_truth = get_ground_truth(q_data)
            print(f"    gt={len(ground_truth)}ch")

            eval_data.append({
                "question"    : question,
                "answer"      : answer,
                "contexts"    : contexts,
                "ground_truth": ground_truth,
                "category"    : q_data["category"],
                "arxiv_id"    : q_data["arxiv_id"],
                "gt_source"   : q_data["gt_source"],
                "intent"      : intent,
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            eval_data.append({
                "question"    : question,
                "answer"      : "",
                "contexts"    : [],
                "ground_truth": "",
                "category"    : q_data["category"],
                "arxiv_id"    : q_data["arxiv_id"],
                "error"       : str(e),
                "intent"      : "",
            })

        time.sleep(SLEEP_BETWEEN)

    valid = len([
        d for d in eval_data
        if d.get("answer") and d.get("contexts")
    ])
    print(f"\n[+] Pipeline queries done: {valid}/{len(questions)} valid")
    return eval_data


# ============================================================
# SECTION 6 — RAGAS Evaluation
# ============================================================

def run_ragas_evaluation(eval_data: list):

    valid_data = [
        d for d in eval_data
        if d.get("answer")
        and d.get("contexts")
        and len(d["contexts"]) > 0
    ]

    if not valid_data:
        print("ERROR: No valid eval data")
        sys.exit(1)

    print(f"\n[+] Running RAGAS on {len(valid_data)} valid queries...")
    print(f"    Skipped  : {len(eval_data) - len(valid_data)} (empty)")
    print(f"    Est. time: ~{len(valid_data) * 2} minutes\n")

    dataset = Dataset.from_list([
        {
            "question"    : d["question"],
            "answer"      : d["answer"],
            "contexts"    : d["contexts"],
            "ground_truth": d.get("ground_truth", ""),
        }
        for d in valid_data
    ])

    # ── instantiate metrics — new RAGAS API requires this ─
    judge_llm = LangchainLLMWrapper(ChatAnthropic(
        model     ="claude-haiku-4-5-20251001",
        api_key   =os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=2000,
    ))

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    results = evaluate(
        dataset        =dataset,
        metrics        =metrics,
        raise_exceptions=False,
    )

    return results, valid_data


# ============================================================
# SECTION 7 — Save Results & Report
# ============================================================

def save_results(
    eval_data    : list,
    ragas_results,
    valid_data   : list,
    output_dir   : Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe_score(val):
        if isinstance(val, list):
            valid = [
                v for v in val
                if v is not None
                and not (isinstance(v, float) and math.isnan(v))
            ]
            return round(sum(valid) / len(valid), 4) if valid else 0.0
        try:
            return round(float(val), 4)
        except Exception:
            return 0.0

    def nan_count(val):
        if isinstance(val, list):
            return sum(
                1 for v in val
                if v is None
                or (isinstance(v, float) and math.isnan(v))
            )
        return 0

    scores = {
        "faithfulness"      : safe_score(ragas_results["faithfulness"]),
        "answer_relevancy"  : safe_score(ragas_results["answer_relevancy"]),
        "context_precision" : safe_score(ragas_results["context_precision"]),
        "context_recall"    : safe_score(ragas_results["context_recall"]),
    }

    nan_counts = {
        "faithfulness"      : nan_count(ragas_results["faithfulness"]),
        "answer_relevancy"  : nan_count(ragas_results["answer_relevancy"]),
        "context_precision" : nan_count(ragas_results["context_precision"]),
        "context_recall"    : nan_count(ragas_results["context_recall"]),
    }

    results_payload = {
        "timestamp"    : datetime.now().isoformat(),
        "collection"   : COLLECTION_NAME,
        "total_queries": len(eval_data),
        "valid_queries": len(valid_data),
        "scores"       : scores,
        "nan_counts"   : nan_counts,
        "per_question" : eval_data,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[+] Results saved: {RESULTS_FILE}")

    def interpret(score):
        if score >= 0.85:   return "excellent"
        elif score >= 0.75: return "good"
        elif score >= 0.65: return "acceptable"
        else:               return "needs improvement"

    # per-category breakdown
    category_scores = {}
    try:
        df = ragas_results.to_pandas()
        df["category"] = [d["category"] for d in valid_data][:len(df)]

        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            category_scores[cat] = {
                "faithfulness"    : round(cat_df["faithfulness"].mean(), 3),
                "answer_relevancy": round(cat_df["answer_relevancy"].mean(), 3),
                "context_precision": round(cat_df["context_precision"].mean(), 3),
                "context_recall"  : round(cat_df["context_recall"].mean(), 3),
                "count"           : len(cat_df),
            }
    except Exception as e:
        print(f"    Category breakdown skipped: {e}")

    # intent breakdown
    intent_counts = {}
    for d in eval_data:
        intent = d.get("intent", "UNKNOWN")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    total_nan    = sum(nan_counts.values())
    eval_quality = (
        "clean" if total_nan == 0
        else f"{total_nan} failed judge calls"
    )

    report = f"""
MULTIMODAL RAG PIPELINE — RAGAS EVALUATION REPORT
{'='*60}
Date              : {datetime.now().strftime('%Y-%m-%d %H:%M')}
Collection        : {COLLECTION_NAME}
Questions (total) : {len(eval_data)}
Questions (valid) : {len(valid_data)}
Embeddings        : CLIP (512-dim) + BGE (768-dim)
Pipeline          : LangGraph (7 nodes)
Query understand  : 3-level (intent → entity → decompose)
Memory            : Two-tier (episodic summary + recent turns)
Judge LLM         : Claude Haiku (2000 max_tokens)
Eval quality      : {eval_quality}

OVERALL SCORES
{'─'*60}
  Faithfulness       : {scores['faithfulness']:.4f}  ({interpret(scores['faithfulness'])})
  Answer Relevancy   : {scores['answer_relevancy']:.4f}  ({interpret(scores['answer_relevancy'])})
  Context Precision  : {scores['context_precision']:.4f}  ({interpret(scores['context_precision'])})
  Context Recall     : {scores['context_recall']:.4f}  ({interpret(scores['context_recall'])})

NaN COUNTS (lower = better)
{'─'*60}
  Faithfulness       : {nan_counts['faithfulness']} failed
  Answer Relevancy   : {nan_counts['answer_relevancy']} failed
  Context Precision  : {nan_counts['context_precision']} failed
  Context Recall     : {nan_counts['context_recall']} failed

INTENT DISTRIBUTION
{'─'*60}"""

    for intent, count in sorted(intent_counts.items()):
        report += f"\n  {intent:<12} : {count} queries"

    if category_scores:
        report += f"\n\nPER-CATEGORY BREAKDOWN\n{'─'*60}"
        for cat, cat_sc in sorted(category_scores.items()):
            report += f"""
  {cat}
    Faithfulness      : {cat_sc['faithfulness']:.3f}
    Answer Relevancy  : {cat_sc['answer_relevancy']:.3f}
    Context Precision : {cat_sc['context_precision']:.3f}
    Context Recall    : {cat_sc['context_recall']:.3f}
    Questions         : {cat_sc['count']}"""

    report += "\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[+] Report saved: {REPORT_FILE}")
    print(report)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MULTIMODAL RAG — RAGAS EVALUATION")
    print("=" * 60)

    # ── check-only mode ───────────────────────────────────
    if "--check-only" in sys.argv:
        check_available_papers(ALL_TEST_QUESTIONS)
        sys.exit(0)

    # ── skip-pipeline mode ────────────────────────────────
    if "--skip-pipeline" in sys.argv:
        print("\n[+] Loading saved pipeline outputs...")
        saved = OUTPUT_DIR / "pipeline_outputs.json"
        if not saved.exists():
            print(f"ERROR: {saved} not found — run without --skip-pipeline first")
            sys.exit(1)
        with open(saved) as f:
            eval_data = json.load(f)
        valid_data = [
            d for d in eval_data
            if d.get("answer") and d.get("contexts")
        ]
        print(f"[+] Loaded {len(eval_data)} saved outputs "
              f"({len(valid_data)} valid)")
        ragas_results, valid_data = run_ragas_evaluation(eval_data)
        save_results(eval_data, ragas_results, valid_data, OUTPUT_DIR)
        sys.exit(0)

    # ── full run ──────────────────────────────────────────

    # Step 1: pre-flight — filter to available papers only
    test_questions = check_available_papers(ALL_TEST_QUESTIONS)

    if not test_questions:
        print("ERROR: No test papers found in collection.")
        print(f"       Make sure '{COLLECTION_NAME}' exists and has data.")
        sys.exit(1)

    # Step 2: run pipeline queries
    eval_data = run_evaluation_queries(test_questions)

    # Step 3: save raw pipeline outputs (allows --skip-pipeline reruns)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "pipeline_outputs.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\n[+] Pipeline outputs saved to eval_output/pipeline_outputs.json")
    print(f"    Tip: rerun RAGAS only with: python evaluate.py --skip-pipeline")

    # Step 4: run RAGAS
    print("\n[+] Starting RAGAS evaluation...")
    ragas_results, valid_data = run_ragas_evaluation(eval_data)

    # Step 5: save results + report
    save_results(eval_data, ragas_results, valid_data, OUTPUT_DIR)

    print("\n[+] Evaluation complete!")
    print(f"    Results : {RESULTS_FILE}")
    print(f"    Report  : {REPORT_FILE}")