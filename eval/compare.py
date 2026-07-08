import argparse
import json
import sys

METRICS = ["hit_at_1", "hit_at_3", "hit_at_5", "mrr"]
EPSILON = 1e-9


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def load_snapshot(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def question_key(q):
    return q.get("id") or q["question"]


def index_by_key(per_question):
    return {question_key(q): q for q in per_question}


def diff_aggregates(baseline, candidate):
    return {m: candidate["aggregate"][m] - baseline["aggregate"][m] for m in METRICS}


def find_flips(baseline, candidate):
    base_by_key = index_by_key(baseline["per_question"])
    cand_by_key = index_by_key(candidate["per_question"])

    hit_to_miss = []
    miss_to_hit = []
    for key, cand_q in cand_by_key.items():
        base_q = base_by_key.get(key)
        if base_q is None:
            continue
        for k in (1, 3, 5):
            field = f"hit_at_{k}"
            if base_q[field] and not cand_q[field]:
                hit_to_miss.append((key, field))
            elif not base_q[field] and cand_q[field]:
                miss_to_hit.append((key, field))
    return hit_to_miss, miss_to_hit


def main():
    parser = argparse.ArgumentParser(description="Diff two eval/run_eval.py snapshots")
    parser.add_argument("baseline", help="Path to the older snapshot JSON")
    parser.add_argument("candidate", help="Path to the newer snapshot JSON")
    parser.add_argument("--threshold", type=float, default=0.0, help="Max allowed metric drop before flagging a regression")
    args = parser.parse_args()

    baseline = load_snapshot(args.baseline)
    candidate = load_snapshot(args.candidate)

    if baseline["config"]["golden_set_count"] != candidate["config"]["golden_set_count"]:
        safe_print(
            f"WARNING: golden set size differs "
            f"({baseline['config']['golden_set_count']} vs {candidate['config']['golden_set_count']}) "
            f"— results may not be directly comparable\n"
        )

    deltas = diff_aggregates(baseline, candidate)

    safe_print(f"Baseline : {args.baseline} ({baseline['timestamp']})")
    safe_print(f"Candidate: {args.candidate} ({candidate['timestamp']})\n")
    safe_print("-" * 60)
    for m in METRICS:
        base_val = baseline["aggregate"][m]
        cand_val = candidate["aggregate"][m]
        delta = deltas[m]
        sign = "+" if delta >= 0 else ""
        safe_print(f"{m:>10}: {base_val:.4f} -> {cand_val:.4f}  ({sign}{delta:.4f})")
    safe_print("-" * 60)

    hit_to_miss, miss_to_hit = find_flips(baseline, candidate)
    if hit_to_miss:
        safe_print("\nHIT -> MISS (regressions):")
        for key, field in hit_to_miss:
            safe_print(f"  - [{field}] {key}")
    if miss_to_hit:
        safe_print("\nMISS -> HIT (improvements):")
        for key, field in miss_to_hit:
            safe_print(f"  - [{field}] {key}")

    regressed = [m for m in METRICS if deltas[m] < -args.threshold - EPSILON]

    safe_print("")
    if regressed:
        safe_print(f"REGRESSION DETECTED: {', '.join(regressed)}")
        sys.exit(1)
    else:
        safe_print("NO REGRESSION")
        sys.exit(0)


if __name__ == "__main__":
    main()
