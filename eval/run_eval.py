"""Small regression eval over the sample documents.

    python -m eval.run_eval
    python -m eval.run_eval --out eval/results/qwen3b.json

It ingests samples/ into a throwaway index, asks every question in
eval/dataset.jsonl through the same QAService the API uses, and scores:

  retrieval   was the passage holding the expected facts shown to the model
  answer      does the answer contain the expected facts (string match)
  cited       does the answer cite the page those facts are on
  grounded    do all citations resolve, and are the cited numbers in the cited passage
  refusal     for unanswerable questions: did it say "not found"

A case passes when the answer is right and it cites the right page.

String matching is a blunt way to judge an answer. It works here because the
expected facts are numbers and fixed terms. It would not work for open
questions, where I would use a second model as a judge and spot-check it.
"""

import argparse
import json
import tempfile
from pathlib import Path

from app.agent.citations import cited_refs, is_not_found
from app.config import Settings
from app.container import build_container
from app.observability.logging import configure_logging

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="write per-question results here")
    args = parser.parse_args()

    configure_logging("WARNING")
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        container = build_container(Settings(data_dir=Path(tmp)))
        for path in sorted((ROOT / "samples").iterdir()):
            container.ingestion.ingest(path.name, path.read_bytes())

        results = [run_case(container, case) for case in cases]
        report = {
            "provider": container.llm.provider,
            "model": container.llm.model,
            "embedding_model": container.settings.embedding_model,
            "retrieval_mode": container.settings.retrieval_mode,
            "prompts": [container.prompts.get(n).tag for n in ("qa_system", "qa_user")],
            "summary": summarise(results),
            "results": results,
        }

    print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


def run_case(container, case: dict) -> dict:
    result = container.qa.ask(case["question"])
    trace = container.traces.get(result.trace_id)
    answerable = case.get("answerable", True)
    row = {
        "id": case["id"],
        "answerable": answerable,
        "answer": result.answer,
        "grounded": result.grounded,
        "latency_ms": result.latency_ms,
        "trace_id": result.trace_id,
        "steps": [s["step"] for s in trace["steps"]],
        "citations_moved": len(trace["citation_repairs"]),
    }

    if not answerable:
        row["refused"] = is_not_found(result.answer)
        row["passed"] = row["refused"]
        return row

    def is_gold(source) -> bool:
        return source["filename"] == case["file"] and source["page"] == case["page"]

    # Checked on the passage text, not just the page number: a page can be split
    # into several chunks and only one of them holds the fact.
    evidence = " ".join(s["text"] for s in trace["sources"] if is_gold(s)).lower()
    answer = result.answer.lower()
    row["retrieval_hit"] = all(fact.lower() in evidence for fact in case["must_contain"])
    row["answer_ok"] = all(fact.lower() in answer for fact in case["must_contain"])
    # What the model cited by itself, before the citation check moved anything.
    model_refs = set(cited_refs(trace["model_answer"]))
    row["cited_gold_by_model"] = any(is_gold(s) for s in trace["sources"] if s["ref"] in model_refs)
    row["cited_gold"] = any(
        s.filename == case["file"] and s.page == case["page"] for s in result.citations
    )
    row["passed"] = row["answer_ok"] and row["cited_gold"]
    return row


def summarise(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    def rate(rows: list[dict], key: str) -> str:
        return f"{sum(bool(r[key]) for r in rows)}/{len(rows)}"

    latencies = sorted(r["latency_ms"] for r in results)
    return {
        "retrieval_hit": rate(answerable, "retrieval_hit"),
        "answer_correct": rate(answerable, "answer_ok"),
        "cites_gold_page_model_alone": rate(answerable, "cited_gold_by_model"),
        "cites_gold_page": rate(answerable, "cited_gold"),
        "grounded": rate(answerable, "grounded"),
        "refusal_correct": rate(unanswerable, "refused"),
        "passed": rate(results, "passed"),
        "answers_with_moved_citation": sum(1 for r in results if r["citations_moved"]),
        "median_latency_ms": latencies[len(latencies) // 2],
    }


def print_report(report: dict) -> None:
    def mark(value) -> str:
        return {True: "yes", False: "NO", None: "-"}[value]

    print(
        f"\nmodel: {report['provider']}/{report['model']}   "
        f"embeddings: {report['embedding_model']}   retrieval: {report['retrieval_mode']}\n"
    )
    print(
        f"{'id':<20}{'retrieved':<11}{'answer':<8}{'cited':<7}{'grounded':<10}{'refused':<9}{'ms':>6}"
    )
    for r in report["results"]:
        print(
            f"{r['id']:<20}"
            f"{mark(r.get('retrieval_hit')):<11}"
            f"{mark(r.get('answer_ok')):<8}"
            f"{mark(r.get('cited_gold')):<7}"
            f"{mark(r['grounded']):<10}"
            f"{mark(r.get('refused')):<9}"
            f"{r['latency_ms']:>6}"
        )
    print()
    for key, value in report["summary"].items():
        print(f"{key:<30}{value}")

    failed = [r for r in report["results"] if not r["passed"]]
    if failed:
        print("\nfailed cases:")
        for r in failed:
            print(f"  {r['id']}: {r['answer'][:160]!r}")


if __name__ == "__main__":
    main()
