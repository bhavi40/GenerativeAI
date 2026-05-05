"""
run.py — Entry point for the RAG pipeline

Usage:
  # single query
  python -m rag_pipeline.run --query "How does attention work?"

  # image query
  python -m rag_pipeline.run --image path/to/diagram.png

  # interactive mode with memory
  python -m rag_pipeline.run
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set.")
    print("Create a .env file with:")
    print("  ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

# these imports are after env check — avoids loading heavy models
# if API key is missing
import anthropic
from rag_pipeline.graph import rag_graph
from rag_pipeline.nodes import ConversationMemory


def run_query(
    query        : str,
    image_path   : str  = None,
    chat_history : list = None,
    extra_state  : dict = None,
) -> dict:
    """
    Run the full RAG pipeline for one query.

    Args:
        query        : user text question
        image_path   : path to uploaded image (optional)
        chat_history : two-tier memory messages from ConversationMemory.to_messages()
        extra_state  : structured state fields (episodic_summary, topics, papers)

    Returns:
        {
          "answer"           : str,
          "sources"          : list,
          "figures"          : list,
          "chat_history"     : list,
          "topics_discussed" : list,
          "papers_referenced": list,
        }
    """
    if not query and not image_path:
        raise ValueError("Provide at least a query or an image")

    initial_state = {
        "query"            : query or "",
        "image_path"       : image_path,
        "chat_history"     : chat_history or [],
        "episodic_summary" : "",
        "topics_discussed" : [],
        "papers_referenced": [],
        "intent"           : "",
        "rewritten_query"  : "",
        "hyde_doc"         : "",
        "query_type"       : "",
        "text_results"     : [],
        "image_results"    : [],
        "ranked_results"   : [],
        "context"          : "",
        "figures"          : [],
        "sources"          : [],
        "answer"           : "",
    }

    # merge structured state if provided
    if extra_state:
        initial_state.update(extra_state)

    result = rag_graph.invoke(initial_state)

    return {
        "answer"           : result.get("answer", ""),
        "sources"          : result.get("sources", []),
        "figures"          : result.get("figures", []),
        "chat_history"     : result.get("chat_history", []),
        "topics_discussed" : result.get("topics_discussed", []),
        "papers_referenced": result.get("papers_referenced", []),
    }


def print_result(result: dict):
    """Pretty print pipeline result to terminal."""
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(result["answer"])

    if result["sources"]:
        print("\n" + "─"*60)
        print("SOURCES")
        print("─"*60)
        for s in result["sources"]:
            print(f"  • {s['paper_title']}")
            print(f"    {s['url']}")

    if result["figures"]:
        print("\n" + "─"*60)
        print(f"RELATED FIGURES ({len(result['figures'])})")
        print("─"*60)
        for f in result["figures"]:
            print(f"  • {f['paper_title']}")
            caption = f.get("caption", "")
            if caption and not caption.startswith("Figure from"):
                print(f"    {caption[:80]}")
            print(f"    {f['image_path']}")

    print("="*60)


def interactive_mode():
    """
    Run in interactive chat mode with two-tier memory.

    Memory flow:
      ConversationMemory holds episodic_summary + recent_turns
      Every turn → add_turn() updates memory
      Every 10 turns → summarize() compresses old turns
      to_messages() builds the message list for the graph
    """
    print("\n" + "="*60)
    print("  Multimodal RAG — AI Research Papers")
    print("  200+ papers | CLIP + BGE | LangGraph + Memory")
    print("="*60)
    print("Commands:")
    print("  Type your question and press Enter")
    print("  'image: path/to/file.png' to query by image")
    print("  'clear' to reset conversation history")
    print("  'quit' to exit")
    print("─"*60)

    # initialized once — persists across all turns in session
    memory     = ConversationMemory()
    raw_client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            memory.reset()
            print("  [conversation cleared]")
            continue

        # parse image query
        query      = ""
        image_path = None
        if user_input.lower().startswith("image:"):
            parts = user_input.split(":", 1)
            if len(parts) == 2:
                image_path = parts[1].strip()
                if not Path(image_path).exists():
                    print(f"Image not found: {image_path}")
                    continue
        else:
            query = user_input

        print("\nSearching...")

        try:
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

            # ── add turn FIRST ────────────────────────────
            memory.add_turn(
                user_msg      = query or f"[image: {image_path}]",
                assistant_msg = result["answer"],
                topics        = result.get("topics_discussed", []),
                papers        = result.get("papers_referenced", []),
            )

            # ── THEN check if summarization threshold hit ─
            if memory.should_summarize():
                memory.summarize(raw_client)

            print_result(result)

            summary_status = "yes" if memory.episodic_summary else "no"
            print(f"\n  [turns: {memory.total_turns} | "
                  f"topics: {len(memory.topics_discussed)} | "
                  f"summary: {summary_status}]")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument("--query", type=str, default="", help="Text query")
    parser.add_argument("--image", type=str, default=None, help="Image path")
    args = parser.parse_args()

    if args.query or args.image:
        print("\nRunning query...")
        result = run_query(
            query      = args.query,
            image_path = args.image,
        )
        print_result(result)
    else:
        interactive_mode()