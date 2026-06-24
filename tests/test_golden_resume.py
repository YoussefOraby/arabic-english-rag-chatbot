"""Tests for synthetic golden resume fixture and evaluation engine.

Verifies:
1. Golden resume PDF is parseable (text extraction via fitz)
2. Evaluation engine evaluate_answer() categorizes failures correctly
3. Golden questions JSON is valid
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz
import pytest

from evaluation.evaluate import evaluate_answer, extract_citations, is_insufficient_data

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def golden_resume_path(data_dir: Path) -> Path:
    return data_dir / "golden_resume.pdf"


@pytest.fixture
def golden_questions_path() -> Path:
    return Path(__file__).resolve().parent.parent / "evaluation" / "golden_questions.json"


@pytest.fixture
def golden_questions(golden_questions_path) -> list[dict]:
    with open(golden_questions_path, encoding="utf-8") as f:
        return json.load(f)


# ── Golden resume PDF parseability ───────────────────────────────────


def test_golden_resume_exists(golden_resume_path: Path):
    assert golden_resume_path.exists(), "golden_resume.pdf not found"
    assert golden_resume_path.stat().st_size > 500


def test_golden_resume_text_extraction(golden_resume_path: Path):
    doc = fitz.open(str(golden_resume_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    assert "Ahmed Hassan" in text, "Name not found in extracted text"
    assert "ahmed@example.com" in text, "Email not found"
    assert "01000000000" in text, "Phone not found"
    assert "Computer Engineering" in text, "Degree not found"
    assert "Arab Academy for Science" in text, "University not found"
    assert "Arabic-English RAG Chatbot" in text, "Project not found"
    assert "RAG" in text, "RAG keyword not found"
    assert len(text) > 500, "Extracted text too short"


# ── Golden questions JSON validity ────────────────────────────────────


class TestGoldenQuestions:
    def test_json_is_valid(self, golden_questions: list[dict]):
        assert len(golden_questions) > 0

    def test_all_have_required_fields(self, golden_questions: list[dict]):
        required = {"id", "question", "should_answer"}
        for q in golden_questions:
            missing = required - set(q.keys())
            assert not missing, f"Question {q.get('id')} missing: {missing}"

    def test_unique_ids(self, golden_questions: list[dict]):
        ids = [q["id"] for q in golden_questions]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    def test_has_resume_and_rag_groups(self, golden_questions: list[dict]):
        groups = {q.get("group", "none") for q in golden_questions}
        assert "synthetic_resume" in groups or any(
            "resume" in q.get("group", "") for q in golden_questions
        ), "No resume group found"
        assert "research_paper" in groups, "No research_paper group found"

    def test_unsupported_questions_exist(self, golden_questions: list[dict]):
        unsupported = [q for q in golden_questions if not q["should_answer"]]
        assert len(unsupported) >= 5, f"Only {len(unsupported)} unsupported questions"

    def test_answerable_resume_questions_have_keywords(self, golden_questions: list[dict]):
        resume = [q for q in golden_questions if "resume" in q.get("group", "").lower()]
        answerable = [q for q in resume if q["should_answer"]]
        for q in answerable:
            has_req = bool(q.get("required_keywords"))
            has_exp = bool(q.get("expected_keywords"))
            assert has_req or has_exp, f"Resume question {q['id']} has no keywords"


# ── evaluate_answer() unit tests ─────────────────────────────────────


class TestEvaluateAnswer:
    def test_passes_correct_answer(self):
        q = {
            "id": "test-001",
            "should_answer": True,
            "required_keywords": ["Naive RAG", "Modular RAG"],
            "forbidden_keywords": [],
            "expected_pages": [1],
        }
        answer = "The paper discusses Naive RAG and Modular RAG"
        result = evaluate_answer("test-001", "?", answer, q)
        assert result["passed"] is True
        assert result["failures"] == []

    def test_false_negative_insufficient_data(self):
        q = {
            "id": "test-002",
            "should_answer": True,
            "required_keywords": ["Naive RAG"],
        }
        answer = "Insufficient data to verify from the uploaded documents."
        result = evaluate_answer("test-002", "?", answer, q)
        assert result["passed"] is False
        assert "FALSE_NEGATIVE" in result["failures"]

    def test_false_positive_answered_unsupported(self):
        q = {
            "id": "test-003",
            "should_answer": False,
            "required_keywords": [],
        }
        answer = "The capital of Japan is Tokyo."
        result = evaluate_answer("test-003", "?", answer, q)
        assert result["passed"] is False
        assert "FALSE_POSITIVE" in result["failures"]

    def test_correct_unsupported_refusal(self):
        q = {
            "id": "test-004",
            "should_answer": False,
            "required_keywords": [],
        }
        answer = "Insufficient data to verify from the uploaded documents."
        result = evaluate_answer("test-004", "?", answer, q)
        assert result["passed"] is True

    def test_missing_keyword_detected(self):
        q = {
            "id": "test-005",
            "should_answer": True,
            "required_keywords": ["RAG", "Modular RAG", "Advanced RAG"],
        }
        answer = "The paper discusses RAG and Advanced RAG"
        result = evaluate_answer("test-005", "?", answer, q)
        assert result["passed"] is False
        assert "MISSING_KEYWORD" in result["failures"]
        assert "Modular RAG" in result.get("missing_keywords", [])

    def test_forbidden_keyword_detected(self):
        q = {
            "id": "test-006",
            "should_answer": True,
            "required_keywords": ["RAG"],
            "forbidden_keywords": ["hallucination"],
        }
        answer = "RAG faces challenges including hallucination"
        result = evaluate_answer("test-006", "?", answer, q)
        assert result["passed"] is False
        assert "FORBIDDEN_KEYWORD" in result["failures"]

    def test_wrong_citation_detected(self):
        q = {
            "id": "test-007",
            "should_answer": True,
            "required_keywords": ["RAG"],
            "expected_pages": [2, 3],
        }
        answer = "RAG is discussed on [page 5]"
        result = evaluate_answer("test-007", "?", answer, q)
        # Citation mismatch is diagnostic, not a primary failure
        assert result["passed"] is True
        assert result.get("citation_detail", {}).get("page_match") is False

    def test_correct_citation_passes(self):
        q = {
            "id": "test-008",
            "should_answer": True,
            "required_keywords": ["RAG"],
            "expected_pages": [2, 3],
        }
        answer = "RAG is discussed on [page 2]"
        result = evaluate_answer("test-008", "?", answer, q)
        assert result["passed"] is True

    def test_empty_answer_fails(self):
        q = {"id": "test-009", "should_answer": True, "required_keywords": []}
        answer = ""
        result = evaluate_answer("test-009", "?", answer, q)
        assert result["passed"] is False

    def test_arabic_keyword_matching(self):
        q = {
            "id": "test-010",
            "should_answer": True,
            "required_keywords": ["RAG"],
        }
        answer = "نظام RAG يعتمد على الاسترجاع والتوليد"
        result = evaluate_answer("test-010", "?", answer, q)
        assert result["passed"] is True
        assert "RAG" in answer

    def test_mixed_language_answer(self):
        q = {
            "id": "test-011",
            "should_answer": True,
            "required_keywords": ["Modular RAG", "Advanced RAG"],
        }
        answer = "Modular RAG هو إطار معياري و Advanced RAG يحسن الأداء"
        result = evaluate_answer("test-011", "?", answer, q)
        assert result["passed"] is True

    def test_expected_keywords_any_match(self):
        q = {
            "id": "test-012",
            "should_answer": True,
            "expected_keywords": ["Ahmed Hassan", "Mohamed Ali"],
            "required_keywords": [],
        }
        answer = "The candidate's name is Ahmed Hassan"
        result = evaluate_answer("test-012", "?", answer, q)
        assert result["passed"] is True

    def test_expected_keywords_no_match(self):
        q = {
            "id": "test-013",
            "should_answer": True,
            "expected_keywords": ["Ahmed Hassan"],
            "required_keywords": [],
        }
        answer = "No name is mentioned in the resume"
        result = evaluate_answer("test-013", "?", answer, q)
        assert result["passed"] is False
        assert "MISSING_EXPECTED_KEYWORD" in result["failures"]

    def test_no_citations_no_fail(self):
        q = {
            "id": "test-014",
            "should_answer": True,
            "required_keywords": ["RAG"],
            "expected_pages": [],
        }
        answer = "RAG is discussed in the paper"
        result = evaluate_answer("test-014", "?", answer, q)
        assert result["passed"] is True

    def test_unexpected_citation_not_fail(self):
        """Citations not in expected_pages should not fail when expected_pages is empty."""
        q = {
            "id": "test-015",
            "should_answer": True,
            "required_keywords": ["RAG"],
            "expected_pages": [],
        }
        answer = "RAG is discussed in the paper [page 5]"
        result = evaluate_answer("test-015", "?", answer, q)
        assert result["passed"] is True


# ── is_insufficient_data unit tests ──────────────────────────────────


class TestIsInsufficientData:
    def test_exact_phrase(self):
        assert is_insufficient_data("Insufficient data to verify from the uploaded documents.")

    def test_cannot_verify(self):
        assert is_insufficient_data("Cannot verify this from the provided documents.")

    def test_not_mentioned(self):
        assert is_insufficient_data("This is not mentioned in the uploaded files.")

    def test_no_info(self):
        assert is_insufficient_data("No information about this is available.")

    def test_i_dont_know(self):
        assert is_insufficient_data("I don't have enough information to answer.")

    def test_real_answer_not_flagged(self):
        assert not is_insufficient_data("RAG stands for Retrieval-Augmented Generation.")

    def test_partial_match_not_confused(self):
        """The word 'not available' without context should still be flagged correctly."""
        assert is_insufficient_data("The requested data is not available in the documents.")


# ── extract_citations unit tests ─────────────────────────────────────


class TestExtractCitations:
    def test_single_citation(self):
        assert extract_citations("See [page 3]") == [3]

    def test_multiple_citations(self):
        assert extract_citations("See [page 2] and [page 5]") == [2, 5]

    def test_bare_number_brackets(self):
        assert extract_citations("See [3]") == [3]

    def test_range_not_broken(self):
        """Range [page 3-5] should still extract the start page."""
        result = extract_citations("See [page 3-5]")
        assert 3 in result

    def test_no_citations(self):
        assert extract_citations("No citations here") == []

    def test_mixed_format(self):
        assert extract_citations("[page 1] and [2]") == [1, 2]

    def test_multiline(self):
        assert extract_citations("First point [1]\nSecond point [2]") == [1, 2]


# ── Citation correctness tests ─────────────────────────────────────────


class TestCitationCorrectness:
    """Ensure citations use real PDF page numbers, not source indices."""

    def test_resume_en_001_passes_with_correct_page(self):
        """resume-en-001 must pass when answer has 'Ahmed Hassan' + [page 1]."""
        q = {
            "id": "resume-en-001",
            "group": "synthetic_resume",
            "should_answer": True,
            "required_keywords": ["Ahmed Hassan"],
            "expected_pages": [1],
            "forbidden_keywords": [],
        }
        answer = "Ahmed Hassan [page 1]"
        result = evaluate_answer("resume-en-001", "What is the candidate's full name?", answer, q)
        assert result["passed"] is True, f"Should pass but got failures: {result['failures']}"

    def test_source_index_not_confused_with_page(self):
        """Using source index [Source 1] as page number should be a diagnostic item.

        If the LLM says [page 2] but expected is page 1, the citation detail
        should show the mismatch, but the answer passes on keywords alone.
        """
        q = {
            "id": "test-citation-001",
            "group": "synthetic_resume",
            "should_answer": True,
            "required_keywords": ["Ahmed Hassan"],
            "expected_pages": [1],
            "forbidden_keywords": [],
        }
        # LLM wrongly cites page 2 (mixing source index with page number)
        answer = "Ahmed Hassan [page 2]"
        result = evaluate_answer("test-citation-001", "What is the name?", answer, q)
        # Answer passes because keyword is correct; citation mismatch is diagnostic
        assert result["passed"] is True, f"Should pass on keywords, got: {result['failures']}"
        diag = result.get("citation_detail", {})
        assert diag.get("page_match") is False, "Citation should not match expected page"

    def test_chunk_header_format_has_page_not_source_index(self):
        """Verify format_chunks_for_context uses 'page:' not '[N]' prefix."""
        from src.rag.prompts import format_chunks_for_context

        class FakeChunk:
            source_file = "golden_resume.pdf"
            page_num = 1
            text = "Ahmed Hassan is a candidate."

        class FakeResult:
            chunk = FakeChunk()
            score = 0.95

        context = format_chunks_for_context([FakeResult()])
        # New format: [Source 1 | file: golden_resume.pdf | page 1]
        assert "page 1" in context, f"Context missing page info: {context[:100]}"
        assert "Source 1" in context, f"Context missing source label: {context[:100]}"
        # The old confusing format [1] should not appear at line start
        assert not context.startswith("[1]"), (
            f"Context still uses old confusing format: {context[:50]}"
        )


# ── Contact field extraction tests ────────────────────────────────────


class TestContactFieldExtraction:
    """Ensure email/phone questions get correct field values, not cross-contamination."""

    def test_resume_en_002_passes_with_email(self):
        """resume-en-002 must pass when answer has 'ahmed@example.com [page 1]'."""
        q = {
            "id": "resume-en-002",
            "group": "synthetic_resume",
            "should_answer": True,
            "required_keywords": ["ahmed@example.com"],
            "expected_pages": [1],
            "forbidden_keywords": [],
        }
        answer = "ahmed@example.com [page 1]"
        result = evaluate_answer("resume-en-002", "What is the email address?", answer, q)
        assert result["passed"] is True, f"Should pass but got failures: {result['failures']}"

    def test_email_question_does_not_return_phone(self):
        """Email question should not return the phone number."""
        q = {
            "id": "test-contact-001",
            "group": "synthetic_resume",
            "should_answer": True,
            "required_keywords": ["ahmed@example.com"],
            "forbidden_keywords": ["01000000000"],
            "expected_pages": [1],
        }
        # LLM wrongly returns phone instead of email
        answer = "01000000000 [page 1]"
        result = evaluate_answer("test-contact-001", "What is the email address?", answer, q)
        assert result["passed"] is False, "Should fail when email returns phone number"
        assert "FORBIDDEN_KEYWORD" in result["failures"] or "MISSING_KEYWORD" in result["failures"]

    def test_phone_question_does_not_return_email(self):
        """Phone question should not return the email address."""
        q = {
            "id": "test-contact-002",
            "group": "synthetic_resume",
            "should_answer": True,
            "required_keywords": ["01000000000"],
            "forbidden_keywords": ["ahmed@example.com"],
            "expected_pages": [1],
        }
        # LLM wrongly returns email instead of phone
        answer = "ahmed@example.com [page 1]"
        result = evaluate_answer("test-contact-002", "What is the phone number?", answer, q)
        assert result["passed"] is False, "Should fail when phone returns email address"
        assert "FORBIDDEN_KEYWORD" in result["failures"] or "MISSING_KEYWORD" in result["failures"]

    def test_extract_structured_field_email(self):
        """_extract_structured_field should find email in chunk text."""
        from src.rag.chain import RAGChain

        class FakeChunk:
            source_file = "golden_resume.pdf"
            page_num = 1
            text = "Ahmed Hassan\nahmed@example.com\n01000000000"
            chunk_id = "fake_001"

        class FakeResult:
            chunk = FakeChunk()
            score = 0.95

        result = RAGChain._extract_structured_field([FakeResult()], "email")
        assert result == "ahmed@example.com", f"Expected email, got: {result}"

    def test_extract_structured_field_phone(self):
        """_extract_structured_field should find phone in chunk text."""
        from src.rag.chain import RAGChain

        class FakeChunk:
            source_file = "golden_resume.pdf"
            page_num = 1
            text = "Ahmed Hassan\nahmed@example.com\n01000000000"
            chunk_id = "fake_002"

        class FakeResult:
            chunk = FakeChunk()
            score = 0.95

        result = RAGChain._extract_structured_field([FakeResult()], "phone")
        assert result == "01000000000", f"Expected phone, got: {result}"

    def test_extract_structured_field_none_for_wrong_type(self):
        """_extract_structured_field should return None for unsupported field type."""
        from src.rag.chain import RAGChain

        result = RAGChain._extract_structured_field([], "name")
        assert result is None

    def test_detect_contact_question_email(self):
        """_detect_contact_question should detect email questions."""
        from src.rag.chain import RAGChain

        assert RAGChain._detect_contact_question("What is the email address?") == "email"
        assert RAGChain._detect_contact_question("What's the e-mail?") == "email"
        assert RAGChain._detect_contact_question("Give me the mail address") == "email"

    def test_detect_contact_question_phone(self):
        """_detect_contact_question should detect phone questions."""
        from src.rag.chain import RAGChain

        assert RAGChain._detect_contact_question("What is the phone number?") == "phone"
        assert RAGChain._detect_contact_question("What's the telephone?") == "phone"
        assert RAGChain._detect_contact_question("Give me the mobile number") == "phone"

    def test_detect_contact_question_none_for_other(self):
        """_detect_contact_question returns None for non-contact questions."""
        from src.rag.chain import RAGChain

        assert RAGChain._detect_contact_question("What is the candidate's full name?") is None
        assert RAGChain._detect_contact_question("What university did they attend?") is None
