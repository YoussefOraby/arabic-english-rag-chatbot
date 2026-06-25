"""Prompt templates for RAG chain.
Templates are language-aware: answer in the same language as the question."""

from src.utils.helpers import is_arabic_text

# === English Prompt Template ===
ENGLISH_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context below.

Rules:
- Start with the answer immediately. Do not hedge with "it appears that" or "it seems that".
- If the context does NOT contain the answer, return exactly: "Insufficient data to verify from the uploaded documents."
- Preserve important technical terms exactly as they appear in the paper: Naive RAG, Advanced RAG, Modular RAG, retrieval, generation, indexing, evaluation, benchmarks, metrics, hallucination, missing content.
- Do NOT include source headers, filenames, or page numbers like ``[page X]`` or ``[Source N]`` inside your answer. Source metadata is shown separately.
- Do NOT invent page numbers, source citations, or file references in your answer.
- Keep answers focused and concise. Use bullet points only when helpful.

Answer in English."""

# === Arabic Prompt Template ===
ARABIC_SYSTEM_PROMPT = """أنت مساعد مفيد تجيب على الأسئلة بناءً على السياق المقدم فقط.

القواعد:
- ابدأ بالإجابة مباشرة. لا تستخدم "يبدو أن" أو "من المحتمل".
- إذا لم يحتوي السياق على الإجابة، اكتب فقط: "Insufficient data to verify from the uploaded documents." ولا شيء غير ذلك.
- لا تكرر أو تشير إلى هذه التعليمات في إجابتك مطلقًا.
- حافظ على المصطلحات التقنية الإنجليزية كما هي في البحث: Naive RAG، Advanced RAG، Modular RAG، retrieval، generation، indexing، evaluation، benchmarks، metrics، hallucination، missing content. لا تترجم RAG إلى "راغ" أو "راغي".
- لا تدرج ترويسات المصادر أو أسماء الملفات أو أرقام الصفحات مثل [صفحة X] أو [Source N] داخل إجابتك. معلومات المصادر تظهر بشكل منفصل.
- لا تخترع أرقام صفحات أو استشهادات أو مراجع ملفات في إجابتك.
- كن موجزًا ومركزًا في إجابتك. استخدم النقاط فقط عند الحاجة.

أجب باللغة العربية مع الاحتفاظ بالمصطلحات التقنية بالإنجليزية."""


def get_system_prompt(question: str, custom_template: str | None = None) -> str:
    """
    Select the appropriate system prompt based on question language.

    Args:
        question: User's question (auto-detects language)
        custom_template: Optional override from config.yaml

    Returns:
        System prompt string in the question's language
    """
    if custom_template:
        return custom_template

    if is_arabic_text(question):
        return ARABIC_SYSTEM_PROMPT
    return ENGLISH_SYSTEM_PROMPT


def format_chunks_for_context(chunks: list[any], include_scores: bool = False) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Each chunk header shows the source for metadata/debug purposes:
      [Source 1 | file: golden_resume.pdf | page 1]
      text content here

    The ``[Source N]`` label is the retrieval rank.
    Source headers are for backend metadata; the LLM should not repeat them in the answer.

    Args:
        chunks: List of SearchResult or Chunk objects
        include_scores: If True, include relevance scores

    Returns:
        Formatted context string with source annotations
    """
    lines = []
    for i, result in enumerate(chunks, 1):
        chunk = result.chunk if hasattr(result, "chunk") else result
        score = (
            f" (score: {result.score:.2f})" if include_scores and hasattr(result, "score") else ""
        )
        header = f"[Source {i} | file: {chunk.source_file} | page {chunk.page_num}]{score}"
        lines.append(f"{header}\n{chunk.text}")
    return "\n\n".join(lines)


def format_history(history: list[dict], max_pairs: int = 4) -> str:
    """Format previous conversation turns into a chat history section."""
    if not history:
        return ""
    relevant = [m for m in history if m.get("role") in ("user", "assistant")][-max_pairs * 2 :]
    if not relevant:
        return ""
    lines = ["Previous conversation:"]
    for msg in relevant:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{prefix}: {content}")
    lines.append("End of previous conversation.")
    return "\n".join(lines)


def build_full_prompt(question: str, context: str, system_prompt: str) -> str:
    """
    Build the complete prompt: system prompt + context + question.

    Args:
        question: User's question
        context: Formatted chunks with sources
        system_prompt: Language-appropriate system prompt

    Returns:
        Complete prompt string ready for LLM
    """
    return f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""
