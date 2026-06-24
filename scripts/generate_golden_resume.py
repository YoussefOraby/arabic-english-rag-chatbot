"""Generate a synthetic golden resume PDF for evaluation & testing.

Output: tests/fixtures/golden_resume.pdf

This PDF contains NO real personal data — all information is synthetic.
"""

from pathlib import Path

import fitz

OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden_resume.pdf"


def build_resume_text() -> str:
    lines = [
        "Ahmed Hassan",
        "ahmed@example.com",
        "01000000000",
        "Senior Machine Learning Engineer",
        "",
        "SUMMARY",
        "Experienced ML engineer with 4+ years building production NLP and recommendation systems.",
        "",
        "SKILLS",
        "Languages: Python, SQL, TypeScript",
        "Frameworks: PyTorch, FastAPI, LangChain, ChromaDB",
        "Tools: Docker, Kubernetes, AWS, Git, CI/CD",
        "",
        "PROJECTS",
        "1. Arabic-English RAG Chatbot: Built a retrieval-augmented generation chatbot supporting Arabic and English with citation verification using ChromaDB and Llama.",
        "2. Churn Prediction Pipeline: Designed end-to-end ML pipeline for customer churn prediction serving 1M+ users with 92% accuracy.",
        "3. Email Classification using Llama: Fine-tuned Llama 2 for multi-label email classification achieving 94% F1 score.",
        "4. Service Desk Ticket Classification: Built automated IT ticket classification system with 89% accuracy using BERT-based models.",
        "",
        "ACHIEVEMENTS",
        "- Won 1st place in university hackathon 2024",
        "- Published research paper on efficient NLP embeddings at EMNLP 2024",
        "- Completed AWS Solutions Architect certification",
        "- Reduced cloud infrastructure costs by 30% at previous company",
        "",
        "EXPERIENCE",
        "Senior ML Engineer | TechCorp (2023-2025)",
        "- Led RAG platform development serving 200+ internal users",
        "- Reduced inference latency by 40% via model quantization",
        "ML Engineer | DataStartup (2021-2023)",
        "- Built recommendation engine handling 10M+ requests/day",
        "",
        "EDUCATION",
        "B.Sc. in Computer Engineering",
        "Arab Academy for Science, Technology and Maritime Transport",
        "Graduated: 2024 | GPA: 3.8/4.0",
    ]
    return "\n".join(lines)


def generate_pdf(output_path: Path) -> None:
    text = build_resume_text()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    rect = fitz.Rect(50, 50, 545, 800)
    page.insert_textbox(rect, text, fontsize=11, fontname="helv", lineheight=1.4)
    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_pdf(OUTPUT)
