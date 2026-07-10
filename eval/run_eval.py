import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from step5_similarity_search import MODEL_PATH, PERSIST_DIR, load_vector_store, search

from eval.judge import JUDGE_LIMITATION, JUDGE_MODEL, aggregate_judge, judge_answer
from eval.metrics import K_MAX, aggregate, find_rank, hit_at_k, is_relevant, reciprocal_rank

EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
RESULTS_DIR = EVAL_DIR / "results"


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def load_golden_set():
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def evaluate_question(vector_store, golden_item, with_judge=False):
    results = search(vector_store, golden_item["question"], k=K_MAX)
    rank = find_rank(results, golden_item)

    record = {
        "id": golden_item.get("id", golden_item["question"]),
        "question": golden_item["question"],
        "expected_source": golden_item["expected_source"],
        "expected_page": golden_item["expected_page"],
        "rank": rank,
        "reciprocal_rank": reciprocal_rank(rank),
        "hit_at_1": hit_at_k(rank, 1),
        "hit_at_3": hit_at_k(rank, 3),
        "hit_at_5": hit_at_k(rank, 5),
        "retrieved": [
            {"source": doc.metadata.get("source"), "page": doc.metadata.get("page"), "score": score}
            for doc, score in results
        ],
    }

    if with_judge:
        from step6_llm_answer import format_context, generate_answer

        answer = generate_answer(golden_item["question"], results)
        context = format_context(results)
        record["answer"] = answer
        record["judge"] = judge_answer(golden_item["question"], answer, context)

    return record


def write_snapshot(per_question, agg, with_judge=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    agg = dict(agg)
    agg["judge"] = aggregate_judge(per_question) if with_judge else None
    snapshot = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "model_path": MODEL_PATH,
            "persist_dir": PERSIST_DIR,
            "k_max": K_MAX,
            "golden_set_count": len(per_question),
            "judge_enabled": with_judge,
            "judge_model": JUDGE_MODEL if with_judge else None,
        },
        "judge_limitation": JUDGE_LIMITATION if with_judge else None,
        "per_question": per_question,
        "aggregate": agg,
    }
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the retrieval (and optionally LLM-judge) eval")
    parser.add_argument("--judge", action="store_true", help="Also generate answers and score them with the LLM judge")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N golden-set questions")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(PERSIST_DIR):
        safe_print(f"No vector store found at '{PERSIST_DIR}'. Run `uv run step4_vector_store.py` first.")
        sys.exit(1)

    golden_set = load_golden_set()
    if args.limit is not None:
        golden_set = golden_set[: args.limit]

    embedding_fn = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    vector_store = load_vector_store(embedding_fn)

    per_question = [evaluate_question(vector_store, item, with_judge=args.judge) for item in golden_set]
    agg = aggregate(per_question)

    safe_print(f"Evaluated {len(per_question)} question(s) against '{PERSIST_DIR}'\n")
    safe_print("-" * 60)
    for q in per_question:
        status = f"rank {q['rank']}" if q["rank"] is not None else "MISS"
        safe_print(f"[{status:>8}] {q['question']}")
        safe_print(f"           expected: {q['expected_source']} p.{q['expected_page']}")
        if args.judge:
            j = q["judge"]
            safe_print(f"           judge: faithfulness={j['faithfulness']} relevance={j['relevance']} completeness={j['completeness']}")
    safe_print("-" * 60)

    safe_print(f"\nHit@1: {agg['hit_at_1']:.4f}")
    safe_print(f"Hit@3: {agg['hit_at_3']:.4f}")
    safe_print(f"Hit@5: {agg['hit_at_5']:.4f}")
    safe_print(f"MRR  : {agg['mrr']:.4f}")

    if args.judge:
        judge_agg = aggregate_judge(per_question)
        safe_print(f"\n!! {JUDGE_LIMITATION}")
        safe_print(f"Judge avg faithfulness: {judge_agg['avg_faithfulness']:.2f}")
        safe_print(f"Judge avg relevance   : {judge_agg['avg_relevance']:.2f}")
        safe_print(f"Judge avg completeness: {judge_agg['avg_completeness']:.2f}")

    out_path = write_snapshot(per_question, agg, with_judge=args.judge)
    safe_print(f"\nSnapshot written to {out_path}")


if __name__ == "__main__":
    main()