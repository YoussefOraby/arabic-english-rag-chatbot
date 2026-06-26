"""Generate a synthetic Arabic labour policy PDF for evaluation & testing.

Output: tests/fixtures/synthetic_arabic_policy.pdf

This PDF contains NO real legal or personal data — all information is synthetic
and derived from publicly known Saudi labour law articles.
"""
from pathlib import Path

import fitz

OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "synthetic_arabic_policy.pdf"


def build_policy_text() -> list[str]:
    return [
        "نظام العمل السعودي",
        "أهم الحقوق والواجبات",
        "",
        "عقد العمل",
        "عقد العمل هو عقد بين صاحب العمل والعامل، يتعهد فيه العامل أن يعمل تحت إدارة صاحب العمل أو إشرافه مقابل أجر.",
        "يجب أن يكتب عقد العمل بنسختين، يحتفظ كل طرف بنسخة.",
        "إذا كان العقد غير مكتوب، يعتبر قائماً ويكون للعامل وحده حق إثباته.",
        "",
        "فترة التجربة",
        "فترة التجربة لا تزيد على 90 يوماً.",
        "ويجوز باتفاق مكتوب بين العامل وصاحب العمل تمديد فترة التجربة بشرط ألا تزيد على 90 يوماً إضافية.",
        "لا يجوز وضع العامل تحت التجربة أكثر من مرة لدى صاحب عمل واحد.",
        "إذا انتهى العقد خلال فترة التجربة، فلا يستحق أي من الطرفين تعويضاً.",
        "",
        "الإجازة السنوية",
        "إجازة سنوية بأجر كامل لا تقل عن 21 يوماً.",
        "وتزداد إلى 30 يوماً إذا أمضى العامل خمس سنوات متصلة في خدمة صاحب العمل.",
        "يجب أن يأخذ العامل إجازته في سنة استحقاقها، ولا يجوز أن يتنازل عنها.",
        "يحق للعامل الحصول على أجره عن أيام الإجازة المستحقة إذا ترك العمل قبل التمتع بها.",
        "",
        "ساعات العمل",
        "لا يجوز تشغيل العامل أكثر من 8 ساعات في اليوم أو 48 ساعة في الأسبوع.",
        "وفي شهر رمضان للمسلمين تخفض ساعات العمل الفعلية إلى 6 ساعات في اليوم أو 36 ساعة في الأسبوع.",
        "لا تدخل فترات الراحة والصلاة والطعام ضمن ساعات العمل الفعلية.",
        "يجوز زيادة ساعات العمل إلى 9 ساعات لبعض فئات العمال أو في بعض الصناعات الخطرة.",
        "",
        "الأجور",
        "يلتزم صاحب العمل بدفع أجور العامل في مواعيدها المحددة.",
        "الأجر الشهري يصرف مرة في الشهر، والأجر اليومي يصرف مرة كل أسبوع.",
        "لا يجوز حسم أكثر من نصف أجر العامل في جميع الأحوال.",
        "",
        "نهاية الخدمة",
        "تنتهي علاقة العمل باتفاق الطرفين أو بانتهاء مدة العقد.",
        "يستحق العامل مكافأة نهاية الخدمة عند انتهاء علاقة العمل.",
        "تحسب المكافأة على أساس أجر نصف شهر عن كل سنة من السنوات الخمس الأولى.",
        "وأجر شهر عن كل سنة من السنوات التالية.",
    ]


def generate_pdf(output_path: Path) -> None:
    lines = build_policy_text()

    doc = fitz.open()
    page = doc.new_page(width=842, height=595)  # Landscape A4
    font = fitz.Font(fontfile="C:/Windows/Fonts/tahoma.ttf")
    tw = fitz.TextWriter(page.rect)

    y = 50
    for line in lines:
        if line == "":
            y += 8
        else:
            # right_to_left=1 stores text in correct visual order
            # PyMuPDF will extract it properly
            tw.append(fitz.Point(50, y), line, font=font, fontsize=11, right_to_left=1)
            y += 7

    page.write_text(writers=tw)
    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_pdf(OUTPUT)
