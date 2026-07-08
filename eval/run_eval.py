import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from step5_similarity_search import MODEL_PATH, PERSIST_DIR, load_vector_store, search

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


def evaluate_question(vector_store, golden_item):
    results = search(vector_store, golden_item["question"], k=K_MAX)
    rank = find_rank(results, golden_item)

    return {
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


def write_snapshot(per_question, agg):
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "model_path": MODEL_PATH,
            "persist_dir": PERSIST_DIR,
            "k_max": K_MAX,
            "golden_set_count": len(per_question),
        },
        "per_question": per_question,
        "aggregate": agg,
    }
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return out_path


def main():
    if not os.path.isdir(PERSIST_DIR):
        safe_print(f"No vector store found at '{PERSIST_DIR}'. Run `uv run step4_vector_store.py` first.")
        sys.exit(1)

    golden_set = load_golden_set()

    embedding_fn = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    vector_store = load_vector_store(embedding_fn)

    per_question = [evaluate_question(vector_store, item) for item in golden_set]
    agg = aggregate(per_question)

    safe_print(f"Evaluated {len(per_question)} question(s) against '{PERSIST_DIR}'\n")
    safe_print("-" * 60)
    for q in per_question:
        status = f"rank {q['rank']}" if q["rank"] is not None else "MISS"
        safe_print(f"[{status:>8}] {q['question']}")
        safe_print(f"           expected: {q['expected_source']} p.{q['expected_page']}")
    safe_print("-" * 60)

    safe_print(f"\nHit@1: {agg['hit_at_1']:.4f}")
    safe_print(f"Hit@3: {agg['hit_at_3']:.4f}")
    safe_print(f"Hit@5: {agg['hit_at_5']:.4f}")
    safe_print(f"MRR  : {agg['mrr']:.4f}")

    out_path = write_snapshot(per_question, agg)
    safe_print(f"\nSnapshot written to {out_path}")


if __name__ == "__main__":
    main()