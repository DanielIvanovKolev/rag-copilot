"""Evaluate retrieval on evals/questions.json (no LLM calls, free and fast)."""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo root, wherever you run the script from
sys.path.insert(0, str(ROOT / "src"))
from retrieve import load_pages, search            # noqa: E402  (import after sys.path change)

PDF_PATH = ROOT / "data" / "RM0090.pdf"
QUESTIONS_PATH = ROOT / "evals" / "questions.json"
RESULTS_DIR = ROOT / "evals" / "results"
STRATEGY = "tfidf"   # change this name when you try a new retrieval strategy
K = 5


def rank_of(expected_page, hits):
    """Position (1-based) of the expected page in the results, or None if it is missing."""
    for i, h in enumerate(hits, start=1):
        if h["page"] == expected_page:
            return i
    return None


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    answerable = [q for q in questions if q["page"] is not None]
    if not answerable:
        print("No questions with an expected page.")
        return

    pages = load_pages(str(PDF_PATH))

    rows = []
    for q in answerable:
        hits = search(q["question"], pages, k=K)
        retrieved = [h["page"] for h in hits]
        rank = rank_of(q["page"], hits)
        rows.append({"question": q["question"], "expected": q["page"], "rank": rank, "retrieved": retrieved})

        status = f"rank {rank}" if rank else "MISS"
        print(f"[{status:>6}] expected p.{q['page']:<5} got {retrieved}  {q['question']}")

    n = len(rows)

    def hit_at(k):
        return sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n

    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    metrics = {"hit@1": hit_at(1), "hit@3": hit_at(3), "hit@5": hit_at(5), "mrr": mrr}

    print(f"\nStrategy: {STRATEGY}, {n} questions, k={K}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = RESULTS_DIR / f"{stamp}_{STRATEGY}.json"
    out.write_text(json.dumps({"strategy": STRATEGY, "k": K, "metrics": metrics, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()