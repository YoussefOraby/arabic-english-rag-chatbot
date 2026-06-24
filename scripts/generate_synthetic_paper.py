"""Generate a synthetic RAG survey paper PDF for isolated evaluation.

Output: tests/fixtures/synthetic_rag_paper.pdf

This is a completely synthetic document. It does NOT reproduce any real paper.
All content is fabricated for testing purposes only.
"""

from pathlib import Path

import fitz

OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "synthetic_rag_paper.pdf"


def build_paper_text() -> str:
    return (
        "Retrieval-Augmented Generation: A Survey of Paradigms and Challenges\n"
        "\n"
        "Abstract. Retrieval-Augmented Generation (RAG) has emerged as a prominent approach\n"
        "for enhancing large language models with external knowledge. This survey categorizes\n"
        "RAG into three paradigms: Naive RAG, Advanced RAG, and Modular RAG. We discuss\n"
        "retrieval methods, evaluation metrics, and open challenges.\n"
        "\n"
        "1 Introduction\n"
        "\n"
        "RAG combines retrieval from external knowledge sources with text generation by LLMs.\n"
        "The core idea is to augment LLMs with relevant retrieved context to improve factual\n"
        "accuracy and reduce hallucination.\n"
        "\n"
        "2 RAG Paradigms\n"
        "\n"
        "2.1 Naive RAG\n"
        "\n"
        "Naive RAG represents the earliest methodology following a retrieve-then-generate\n"
        "pipeline. It consists of three stages: indexing, retrieval, and generation. The\n"
        "indexing stage pre-processes documents into chunks and stores them in a vector\n"
        "database. The retrieval stage finds relevant chunks using embedding similarity. The\n"
        "generation stage produces an answer conditioned on the retrieved context. Naive RAG\n"
        "features a simple chain-like structure with no feedback between stages. Limitations\n"
        "include limited retrieval quality, no iterative refinement, and sensitivity to chunk\n"
        "quality. The sequential nature means errors in retrieval propagate to generation.\n"
        "\n"
        "2.2 Advanced RAG\n"
        "\n"
        "Advanced RAG introduces targeted optimizations before and after retrieval to improve\n"
        "quality. Pre-retrieval strategies include query rewriting, query expansion, and chunk\n"
        "optimization. Post-retrieval strategies include reranking, context filtering, and\n"
        "answer verification. These additions make the pipeline more robust than the basic\n"
        "Naive RAG approach. Advanced RAG addresses many limitations of Naive RAG by adding\n"
        "feedback loops and quality gates. Query rewriting reformulates ambiguous questions,\n"
        "while reranking ensures only the most relevant passages are passed to the LLM.\n"
        "\n"
        "2.3 Modular RAG\n"
        "\n"
        "Modular RAG adopts a modular architecture with trainable and flexible components.\n"
        "It enables task-specific customization and pattern recombination. Different modules\n"
        "can be swapped, added, or removed depending on the application. The modular design\n"
        "supports iterative retrieval where the system alternates between retrieval and\n"
        "generation, recursive retrieval for hierarchical knowledge access, and adaptive\n"
        "retrieval where the system decides when to retrieve. Modular RAG represents the\n"
        "most flexible paradigm, allowing fine-grained control over the retrieval-generation\n"
        "loop.\n"
        "\n"
        "3 Retrieval Methods\n"
        "\n"
        "3.1 Dense Retrieval\n"
        "\n"
        "Dense retrieval uses embedding models to map queries and documents to a shared vector\n"
        "space. Neural network encoders produce dense representations that capture semantic\n"
        "similarity beyond exact keyword matching. Popular models include sentence transformers\n"
        "and biencoder architectures.\n"
        "\n"
        "3.2 Sparse Retrieval (BM25)\n"
        "\n"
        "BM25 is a traditional bag-of-words retrieval method based on term frequency and\n"
        "inverse document frequency. It remains competitive, especially for exact keyword\n"
        "matching scenarios. BM25 does not require any training data and works well out of\n"
        "the box.\n"
        "\n"
        "3.3 Hybrid Retrieval\n"
        "\n"
        "Hybrid retrieval combines dense and sparse methods to leverage their complementary\n"
        "strengths. Dense retrieval captures semantic similarity while BM25 handles exact\n"
        "term matching. The scores are typically combined via reciprocal rank fusion or\n"
        "weighted averaging.\n"
        "\n"
        "3.4 Reranking\n"
        "\n"
        "Reranking applies a cross-encoder model to re-score the top-k candidates from an\n"
        "initial retrieval step. Cross-encoders jointly encode query-document pairs, yielding\n"
        "more accurate relevance judgments. This two-stage approach balances efficiency and\n"
        "effectiveness.\n"
        "\n"
        "4 Challenges\n"
        "\n"
        "4.1 Missing Content\n"
        "\n"
        "The retrieved documents may lack the specific information needed to answer the\n"
        "question. This is the most fundamental challenge in RAG systems.\n"
        "\n"
        "4.2 Hallucination\n"
        "\n"
        "The LLM may generate information not present in the retrieved context. Even when\n"
        "relevant context is provided, the model might ignore it or fabricate details.\n"
        "\n"
        "4.3 Relevance Mismatch\n"
        "\n"
        "Retrieved chunks may be topically related but fail to contain the specific answer.\n"
        "The retriever picks broadly relevant passages that miss the precise information\n"
        "required.\n"
        "\n"
        "5 Evaluation Metrics\n"
        "\n"
        "Evaluation of RAG systems can be categorized into three dimensions: retrieval\n"
        "quality, generation quality, and end-to-end performance. Common metrics include\n"
        "accuracy, precision, recall, and F1 for answer correctness. Faithfulness metrics\n"
        "measure whether the generated answer is supported by the retrieved context. BLEU\n"
        "and ROUGE evaluate lexical overlap between generated and reference answers.\n"
        "\n"
        "6 Conclusion\n"
        "\n"
        "RAG continues to evolve with new paradigms, improved retrieval methods, and better\n"
        "evaluation frameworks. The modular paradigm offers the most flexibility for future\n"
        "innovation.\n"
    )


def generate_pdf(output_path: Path) -> None:
    full_text = build_paper_text()
    lines = full_text.split("\n")
    n = len(lines)

    doc = fitz.open()
    rect = fitz.Rect(50, 50, 545, 800)
    page = doc.new_page(width=595, height=842)

    # Find how many lines fit by binary search
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        test_doc = fitz.open()
        test_page = test_doc.new_page(width=595, height=842)
        chunk = "\n".join(lines[:mid])
        ret = test_page.insert_textbox(rect, chunk, fontsize=9, fontname="helv", lineheight=1.2)
        test_doc.close()
        if ret >= 0:
            lo = mid
        else:
            hi = mid - 1

    # Write first page
    chunk = "\n".join(lines[:lo])
    page.insert_textbox(rect, chunk, fontsize=9, fontname="helv", lineheight=1.2)

    # Remaining lines go to page 2
    remaining = lines[lo:]
    if remaining:
        page2 = doc.new_page(width=595, height=842)
        page2.insert_textbox(rect, "\n".join(remaining), fontsize=9, fontname="helv", lineheight=1.2)

    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path} ({output_path.stat().st_size} bytes, lines={n})")


if __name__ == "__main__":
    generate_pdf(OUTPUT)
