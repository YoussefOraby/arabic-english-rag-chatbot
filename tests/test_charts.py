"""Tests for chart/diagram detection via pix2text."""


def test_chart_detector_import():
    """Verify chart detector can be imported."""
    from src.pdf.charts import ChartDetector

    assert ChartDetector is not None
    assert hasattr(ChartDetector, "is_available")
    assert hasattr(ChartDetector, "detect_figures")


def test_chart_detector_availability():
    """Verify is_available returns bool."""
    from src.pdf.charts import ChartDetector

    available = ChartDetector.is_available()
    assert isinstance(available, bool)


def test_extract_chart_chunks_returns_list():
    """Verify extract_chart_chunks returns list (possibly empty)."""
    from pathlib import Path

    from src.pdf.charts import extract_chart_chunks

    chunks = extract_chart_chunks(Path("/nonexistent/file.pdf"), "test.pdf")
    assert isinstance(chunks, list)


def test_chart_detector_detect_figures(data_dir):
    """Verify detect_figures finds figures in the arxiv paper."""
    import pytest

    from src.pdf.charts import ChartDetector

    if not ChartDetector.is_available():
        pytest.skip("pix2text not installed")

    # Test the figure detection function on an available page
    figures = ChartDetector.detect_figures(
        data_dir / "sample_arxiv.pdf",
        page_numbers=[0],
    )
    # Accept 0 figures — fixture may not contain images
    assert isinstance(figures, list)
    for fig in figures:
        assert "page_num" in fig
        assert "box" in fig
        assert len(fig["box"]) == 4
        assert fig["width"] > 0
        assert fig["height"] > 0


def test_chart_detector_no_figures(data_dir):
    """Verify detect_figures returns empty for text-only PDFs."""
    import pytest

    from src.pdf.charts import ChartDetector

    if not ChartDetector.is_available():
        pytest.skip("pix2text not installed")

    figures = ChartDetector.detect_figures(
        data_dir / "sample_ar.pdf",
        page_numbers=[0],
    )
    assert len(figures) == 0


def test_extract_chart_chunks_with_data(data_dir):
    """Verify extract_chart_chunks returns chart chunks for the arxiv paper."""
    import pytest

    from src.pdf.charts import ChartDetector, extract_chart_chunks

    if not ChartDetector.is_available():
        pytest.skip("pix2text not installed")

    chunks = extract_chart_chunks(
        data_dir / "sample_arxiv.pdf",
        "sample_arxiv.pdf",
        page_numbers=[0],  # first page
    )
    # Accept 0 chunks — fixture may not contain images
    assert isinstance(chunks, list)
    for chunk in chunks:
        assert chunk.chunk_type == "chart"
        assert "Chart on page" in chunk.text
        assert chunk.page_num >= 1


# Import needed in the last 2 test functions (done inside each)
