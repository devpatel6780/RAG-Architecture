from types import SimpleNamespace

from eval.metrics import aggregate, find_rank, hit_at_k, is_relevant, reciprocal_rank


def make_doc(source, page=None):
    return SimpleNamespace(metadata={"source": source, "page": page})


def make_golden(expected_source, expected_page):
    return {"expected_source": expected_source, "expected_page": expected_page}


def test_is_relevant_matches_source_and_page():
    doc = make_doc("data/Info_Document.pdf", page=19)
    golden = make_golden("data/Info_Document.pdf", 19)
    assert is_relevant(doc, golden) is True


def test_is_relevant_wrong_page():
    doc = make_doc("data/Info_Document.pdf", page=5)
    golden = make_golden("data/Info_Document.pdf", 19)
    assert is_relevant(doc, golden) is False


def test_is_relevant_wrong_source():
    doc = make_doc("data/sample.txt", page=None)
    golden = make_golden("data/Info_Document.pdf", 19)
    assert is_relevant(doc, golden) is False


def test_is_relevant_null_expected_page_matches_any_page():
    doc = make_doc("data/sample.txt", page=None)
    golden = make_golden("data/sample.txt", None)
    assert is_relevant(doc, golden) is True


def test_find_rank_returns_1_indexed_position():
    golden = make_golden("data/Info_Document.pdf", 19)
    results = [
        (make_doc("data/Info_Document.pdf", page=5), 0.5),
        (make_doc("data/Info_Document.pdf", page=19), 0.9),
    ]
    assert find_rank(results, golden) == 2


def test_find_rank_returns_none_on_miss():
    golden = make_golden("data/Info_Document.pdf", 19)
    results = [(make_doc("data/Info_Document.pdf", page=5), 0.5)]
    assert find_rank(results, golden) is None


def test_hit_at_k():
    assert hit_at_k(1, 1) is True
    assert hit_at_k(3, 1) is False
    assert hit_at_k(3, 3) is True
    assert hit_at_k(None, 5) is False


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_aggregate_empty():
    agg = aggregate([])
    assert agg["num_questions"] == 0
    assert agg["hit_at_1"] == 0.0
    assert agg["mrr"] == 0.0


def test_aggregate_averages_across_questions():
    per_question = [
        {"hit_at_1": True, "hit_at_3": True, "hit_at_5": True, "reciprocal_rank": 1.0},
        {"hit_at_1": False, "hit_at_3": False, "hit_at_5": True, "reciprocal_rank": 0.2},
    ]
    agg = aggregate(per_question)
    assert agg["num_questions"] == 2
    assert agg["hit_at_1"] == 0.5
    assert agg["hit_at_3"] == 0.5
    assert agg["hit_at_5"] == 1.0
    assert agg["mrr"] == 0.6
