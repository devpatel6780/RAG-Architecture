K_MAX = 5


def is_relevant(doc, golden_item):
    if doc.metadata.get("source") != golden_item["expected_source"]:
        return False
    if golden_item["expected_page"] is None:
        return True
    return doc.metadata.get("page") == golden_item["expected_page"]


def find_rank(results, golden_item):
    # results: list[(Document, score)] from step5's search(), best match first
    for i, (doc, _score) in enumerate(results):
        if is_relevant(doc, golden_item):
            return i + 1
    return None


def hit_at_k(rank, k):
    return rank is not None and rank <= k


def reciprocal_rank(rank):
    return 1.0 / rank if rank is not None else 0.0


def aggregate(per_question):
    n = len(per_question)
    if n == 0:
        return {"num_questions": 0, "hit_at_1": 0.0, "hit_at_3": 0.0, "hit_at_5": 0.0, "mrr": 0.0}
    return {
        "num_questions": n,
        "hit_at_1": sum(q["hit_at_1"] for q in per_question) / n,
        "hit_at_3": sum(q["hit_at_3"] for q in per_question) / n,
        "hit_at_5": sum(q["hit_at_5"] for q in per_question) / n,
        "mrr": sum(q["reciprocal_rank"] for q in per_question) / n,
    }
