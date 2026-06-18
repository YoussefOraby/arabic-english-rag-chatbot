"""Tests for evaluation framework (scoring, matching, refusal detection)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import (
    _ARABIC_ENGLISH_EQUIVALENTS,
    _keyword_matches_text,
    check_required_keywords,
    check_unsupported_accuracy,
    cosine_similarity,
    keyword_match_score,
    normalize_arabic,
)

# ── Arabic Normalization ────────────────────────────────────────────


def test_normalize_arabic_removes_diacritics():
    assert normalize_arabic("كِتَاب") == "كتاب"


def test_normalize_arabic_unifies_alef():
    assert normalize_arabic("أحمد") == "احمد"
    assert normalize_arabic("إبراهيم") == "ابراهيم"
    assert normalize_arabic("آدم") == "ادم"


def test_normalize_arabic_yah():
    assert normalize_arabic("على") == "علي"


def test_normalize_arabic_teh_marbuta():
    assert normalize_arabic("مدرسة") == "مدرسه"


def test_normalize_arabic_no_latin_effect():
    assert normalize_arabic("hello") == "hello"


# ── Keyword Variants ────────────────────────────────────────────────


def test_keyword_matches_basic():
    assert _keyword_matches_text("flexible", "it is flexible")


def test_keyword_matches_ity_suffix():
    assert _keyword_matches_text("flexible", "it provides flexibility")


def test_keyword_matches_ing_suffix():
    assert _keyword_matches_text("retrieve", "how retrieval works")


def test_keyword_matches_tion_suffix():
    assert _keyword_matches_text("generate", "the generation process")


# ── Keyword Matching ────────────────────────────────────────────────


def test_keyword_matches_exact():
    assert _keyword_matches_text("Naive RAG", "this discusses naive rag")


def test_keyword_matches_morphological():
    assert _keyword_matches_text("flexible", "it offers flexibility")


def test_keyword_matches_arabic_equivalent():
    for equiv in _ARABIC_ENGLISH_EQUIVALENTS.get("challenges", []):
        if "التحديات" in equiv:
            assert _keyword_matches_text("challenges", equiv)


def test_keyword_matches_no_false_positive():
    assert not _keyword_matches_text("BM25", "This is about BM35")


def test_keyword_not_in_answer():
    assert not _keyword_matches_text("hallucination", "the paper discusses retrieval")


# ── Arabic-English Equivalents ──────────────────────────────────────


def test_arabic_equivalent_RAG():
    assert _keyword_matches_text("RAG", "نظام راغ متكامل")


def test_arabic_equivalent_challenges():
    assert _keyword_matches_text("challenges", "التحديات")


def test_arabic_equivalent_naive_rag():
    assert _keyword_matches_text("Naive RAG", "راغ العادي")


def test_arabic_equivalent_advanced_rag():
    assert _keyword_matches_text("Advanced RAG", "راغ المتقدم")


def test_arabic_equivalent_retrieval():
    assert _keyword_matches_text("retrieval", "استرجاع")


# ── check_required_keywords (integrated) ────────────────────────────


def test_check_required_keywords_all_present():
    answer = "Naive RAG and Advanced RAG and Modular RAG"
    assert check_required_keywords(answer, ["Naive RAG", "Advanced RAG"]) is True


def test_check_required_keywords_none():
    assert check_required_keywords("hello", None) is None


def test_check_required_keywords_empty():
    assert check_required_keywords("hello", []) is None


def test_check_required_keywords_missing():
    assert check_required_keywords("Naive RAG only", ["Modular RAG"]) is False


def test_check_required_keywords_morphological():
    answer = "it provides flexibility"
    assert check_required_keywords(answer, ["flexible"]) is True


def test_check_required_keywords_arabic():
    answer = "التحديات التي تواجه RAG"
    assert check_required_keywords(answer, ["challenges"]) is True


def test_check_required_keywords_case_insensitive():
    assert check_required_keywords("HELLO WORLD", ["hello"]) is True


def test_check_required_keywords_substring():
    assert check_required_keywords("the RAG framework", ["RAG"]) is True


def test_check_required_keywords_mixed_arabic_rag():
    answer = "راغ العادي وراغ المتقدم"
    assert check_required_keywords(answer, ["Naive RAG", "Advanced RAG"]) is True


def test_check_required_keywords_arabic_modular():
    answer = "نظام راغ متكامل لتحسين الأداء"
    assert check_required_keywords(answer, ["Modular RAG"]) is True


# ── Unsupported Question Accuracy ──────────────────────────────────


def test_unsupported_insufficient_data_flag():
    result = {"insufficient_data": True, "answer": "", "citations": [], "source_documents": {}}
    assert check_unsupported_accuracy(result) is True


def test_unsupported_insufficient_data_phrase():
    result = {
        "insufficient_data": False,
        "answer": "Insufficient data to verify from the uploaded documents.",
        "citations": [],
        "source_documents": {},
    }
    assert check_unsupported_accuracy(result) is True


def test_unsupported_i_dont_know_based_on_docs():
    result = {
        "insufficient_data": False,
        "answer": "I don't know based on the provided documents.",
        "citations": [],
        "source_documents": {},
    }
    assert check_unsupported_accuracy(result) is True


def test_unsupported_no_mention():
    result = {
        "insufficient_data": False,
        "answer": "There is no mention of the 2022 FIFA World Cup in the given context.",
        "citations": [],
        "source_documents": {},
    }
    assert check_unsupported_accuracy(result) is True


def test_unsupported_arabic_refusal():
    result = {
        "insufficient_data": False,
        "answer": "لا أعرف بناءً على المستندات المقدمة",
        "citations": [],
        "source_documents": {},
    }
    assert check_unsupported_accuracy(result) is True


def test_unsupported_answer_with_citations_fails():
    result = {
        "insufficient_data": False,
        "answer": "I don't know based on the provided documents.",
        "citations": [{"pages": [1], "verified": True}],
        "source_documents": {},
    }
    assert check_unsupported_accuracy(result) is False


def test_unsupported_answer_with_sources_passes():
    """Source_documents are always present (retriever runs), not a sign of answering."""
    result = {
        "insufficient_data": False,
        "answer": "I don't know based on the provided documents.",
        "citations": [],
        "source_documents": {"doc.pdf": [1]},
    }
    assert check_unsupported_accuracy(result) is True


def test_unsupported_real_answer_fails():
    result = {
        "insufficient_data": False,
        "answer": "Argentina won the 2022 FIFA World Cup.",
        "citations": [],
        "source_documents": {},
    }
    assert check_unsupported_accuracy(result) is False


# ── Existing functionality still works ──────────────────────────────


def test_keyword_match_score():
    assert keyword_match_score("Naive RAG Advanced RAG", "Naive RAG Advanced RAG") == 1.0


def test_keyword_match_score_partial():
    score = keyword_match_score("hello world foo", "hello world bar")
    assert 0.5 < score < 1.0


def test_keyword_match_score_no_match():
    assert keyword_match_score("abc", "xyz") == 0.0


def test_cosine_similarity_identical():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
