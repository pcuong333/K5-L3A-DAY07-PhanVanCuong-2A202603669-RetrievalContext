"""Run the Day 7 hybrid agent with a locally configured DeepSeek key."""
import argparse
import json
import time
from pathlib import Path

from agentic import run_agent
from hybrid_retriever import HybridRetriever
from openrouter_controller import DeepSeekController


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "day6" / "questions.json"
CHUNKS = ROOT / "data" / "day6" / "converted" / "chunks.json"
OUTPUT = ROOT / "starter" / "results" / "deepseek-hybrid-dev"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.budget < 1 or args.max_searches < 1:
        raise ValueError("budget and max-searches must be greater than zero")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    controller = DeepSeekController()
    retriever = HybridRetriever(chunks, strategy="B")
    output = args.output or ROOT / "starter" / "results" / f"deepseek-hybrid-{args.split}-b{args.budget}"
    traces = []
    for question in questions:
        if question.get("split") != args.split:
            continue
        started = time.perf_counter()
        try:
            result = run_agent(
                question["question"],
                controller,
                retriever,
                filters=question.get("filter"),
                max_searches=args.max_searches,
                top_k=3,
                budget=args.budget,
            )
            result["query_id"] = question["id"]
        except (ConnectionError, RuntimeError, TimeoutError) as exc:
            result = {
                "query_id": question["id"],
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [{"event": "error", "error": str(exc)}],
            }
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        traces.append(result)
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"split": args.split, "traces": len(traces), "output": str(output)}))


if __name__ == "__main__":
    main()
