"""Download and generate sample PDFs for testing.
Run: python scripts/download_samples.py
Output: data/raw/sample_arxiv.pdf + data/raw/sample_ar.pdf"""

from pathlib import Path

# Ensure data/raw/ exists
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# === Step 1: Download one arXiv paper (English) ===
def download_arxiv_paper():
    """Download a real English PDF from arXiv (Arabic NLP topic)."""
    pdf_path = DATA_DIR / "sample_arxiv.pdf"
    if pdf_path.exists():
        print(f"[OK] arXiv paper already exists: {pdf_path}")
        return pdf_path

    arxiv_id = "2305.12345"  # Example: Arabic NLP paper on arXiv
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    print(f"[*] Downloading arXiv paper {arxiv_id}...")
    try:
        import urllib.request

        urllib.request.urlretrieve(url, pdf_path)
        print(f"[OK] Downloaded: {pdf_path}")
    except Exception as e:
        print(f"[!] Download failed: {e}")
        print("[*] Generating fallback English PDF instead...")
        generate_fallback_english_pdf(pdf_path)

    return pdf_path


def generate_fallback_english_pdf(pdf_path: Path):
    """Create a simple English PDF if arXiv download fails."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=12)

    text = """Natural Language Processing for Arabic: Challenges and Opportunities

Arabic is a Semitic language spoken by over 400 million people worldwide.
It has a rich morphological system and complex word formation rules.
Modern Standard Arabic differs significantly from dialectal varieties.
This paper reviews recent advances in Arabic NLP including machine translation,
sentiment analysis, and named entity recognition.

Key challenges include:
- Lack of large annotated datasets
- Arabic script ambiguity (missing diacritics)
- Dialectal variation across regions
- Code-switching between Arabic and other languages

Recent transformer-based models have shown promising results for Arabic.
Multilingual models like mBERT and XLM-R perform well on standard benchmarks.
However, they still struggle with dialectal Arabic and code-mixed text."""

    pdf.multi_cell(0, 6, text)
    pdf.output(str(pdf_path))
    print(f"[OK] Created fallback English PDF: {pdf_path}")


# === Step 2: Generate Arabic PDF ===
def generate_arabic_pdf():
    """Create an Arabic PDF using fpdf2 with Noto Sans Arabic font."""
    pdf_path = DATA_DIR / "sample_ar.pdf"
    if pdf_path.exists():
        print(f"[OK] Arabic PDF already exists: {pdf_path}")
        return pdf_path

    # Download Noto Sans Arabic font if not present
    font_dir = Path(__file__).resolve().parent.parent / "fonts"
    font_dir.mkdir(exist_ok=True)
    font_path = font_dir / "NotoSansArabic-Regular.ttf"

    if not font_path.exists():
        print("[*] Downloading Noto Sans Arabic font...")
        try:
            import urllib.request

            url = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansArabic/full/ttf/NotoSansArabic-Regular.ttf"
            urllib.request.urlretrieve(url, font_path)
            print(f"[OK] Font downloaded: {font_path}")
        except Exception as e:
            print(f"[!] Font download failed: {e}")
            print("[*] Arabic PDF will use built-in Helvetica (limited Arabic support)")
            font_path = None

    try:
        from fpdf import FPDF

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()

        if font_path and font_path.exists():
            pdf.add_font("NotoArabic", "", str(font_path))
            pdf.set_font("NotoArabic", size=16)
        else:
            # Helvetica has limited Arabic but works as fallback
            pdf.set_font("Helvetica", size=14)

        arabic_text = """الذكاء الاصطناعي وتطبيقاته في معالجة اللغة العربية

شهد مجال معالجة اللغة الطبيعية تطورا كبيرا في السنوات الأخيرة، خاصة مع ظهور نماذج التعلم العميق والمحولات.
تواجه معالجة اللغة العربية تحديات فريدة بسبب تعقيدها الصرفي والنحوي.

من أبرز التحديات:
- تعدد اللهجات العربية واختلافها عن اللغة العربية الفصحى
- نقص البيانات المصنفة عالية الجودة
- تعقيد النظام الصرفي للغة العربية
- ظاهرة تعدد المعاني واعتمادها على السياق

على الرغم من هذه التحديات، فقد حققت النماذج الحديثة نتائج واعدة في مجالات متعددة مثل:
- الترجمة الآلية بين العربية والإنجليزية
- تحليل المشاعر للنصوص العربية
- التعرف على الكيانات المسماة
- الإجابة عن الأسئلة في النصوص العربية"""

        pdf.multi_cell(0, 10, arabic_text, new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(pdf_path))
        print(f"[OK] Created Arabic PDF: {pdf_path}")
    except Exception as e:
        print(f"[!] Failed to create Arabic PDF: {e}")

    return pdf_path


# === Main ===
if __name__ == "__main__":
    print("=" * 50)
    print("[DOWNLOAD] Downloading/Generating Sample PDFs")
    print("=" * 50)

    # Step 1: English arXiv paper
    print("\n[1/2] English PDF (arXiv)")
    try:
        download_arxiv_paper()
    except Exception as e:
        print(f"[!] Error downloading arXiv: {e}")

    # Step 2: Arabic PDF
    print("\n[2/2] Arabic PDF")
    try:
        generate_arabic_pdf()
    except Exception as e:
        print(f"[!] Error generating Arabic PDF: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("[FILES] Sample PDFs in:", DATA_DIR)
    for f in sorted(DATA_DIR.glob("*.pdf")):
        size = f.stat().st_size / 1024
        print(f"   {f.name} ({size:.1f} KB)")
    print("=" * 50)
