import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import get_settings
from judge import Judge, create_judge
from metrics import compute_citation_accuracy, compute_hit, compute_mrr

EVAL_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the PDF RAG backend against a golden QA dataset."
    )
    parser.add_argument("--dataset", default=str(EVAL_DIR / "golden_qa.json"))
    parser.add_argument("--api-base-url", default="http://localhost:9099")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


async def evaluate_item(
    client: httpx.AsyncClient, judge: Judge, item: dict, top_k: int
) -> dict:
    golden_pages = {(p["document"], p["page"]) for p in item["golden_pages"]}

    try:
        response = await client.post(
            "/api/chat", json={"question": item["question"], "top_k": top_k}
        )
        response.raise_for_status()
        data = response.json()
        retrieved_pages = [
            (source["document"], source["page"]) for source in data["sources"]
        ]
        judge_result = await judge.score_correctness(
            item["question"], item["reference_answer"], data["answer"]
        )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {
            "id": item["id"],
            "question": item["question"],
            "error": str(exc),
            "hit": False,
            "rank": None,
            "mrr": 0.0,
            "citation_accuracy": 0.0,
            "correctness_score": 0.0,
            "correctness_reasoning": "",
            "generated_answer": "",
            "retrieved_pages": [],
        }

    rank = next(
        (i + 1 for i, page in enumerate(retrieved_pages) if page in golden_pages), None
    )

    return {
        "id": item["id"],
        "question": item["question"],
        "hit": compute_hit(retrieved_pages, golden_pages),
        "rank": rank,
        "mrr": compute_mrr(retrieved_pages, golden_pages),
        "citation_accuracy": compute_citation_accuracy(retrieved_pages, golden_pages),
        "correctness_score": judge_result.score,
        "correctness_reasoning": judge_result.reasoning,
        "generated_answer": data["answer"],
        "retrieved_pages": retrieved_pages,
    }


def summarize(per_query: list[dict]) -> dict:
    n = len(per_query)
    return {
        "num_queries": n,
        "hit_rate": sum(1 for r in per_query if r["hit"]) / n,
        "mean_mrr": sum(r["mrr"] for r in per_query) / n,
        "mean_citation_accuracy": sum(r["citation_accuracy"] for r in per_query) / n,
        "mean_correctness": sum(r["correctness_score"] for r in per_query) / n,
    }


async def run(args: argparse.Namespace) -> dict:
    settings = get_settings()
    judge = create_judge(settings)
    golden_items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    async with httpx.AsyncClient(base_url=args.api_base_url, timeout=300) as client:
        per_query = [
            await evaluate_item(client, judge, item, args.top_k) for item in golden_items
        ]

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "top_k": args.top_k,
        "per_query": per_query,
        "aggregates": summarize(per_query),
    }


def write_and_report(result: dict, output_arg: str | None) -> Path:
    output_path = (
        Path(output_arg)
        if output_arg
        else EVAL_DIR / "results" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nResults written to {output_path}\n")
    print(f"{'metric':<25}{'value':>10}")
    for key, value in result["aggregates"].items():
        line_value = f"{value:>10.3f}" if isinstance(value, float) else f"{value:>10}"
        print(f"{key:<25}{line_value}")

    return output_path


async def main() -> None:
    args = parse_args()
    result = await run(args)
    write_and_report(result, args.output)


if __name__ == "__main__":
    asyncio.run(main())
