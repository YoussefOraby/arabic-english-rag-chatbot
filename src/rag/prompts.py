"""Prompt templates for RAG chain.
Templates are language-aware: answer in the same language as the question."""

from src.utils.helpers import is_arabic_text

# === English Prompt Template ===
ENGLISH_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context below.

Rules:
- Start with the answer immediately. Do not hedge with "it appears that" or "it seems that".
- If the context does NOT contain the answer, return exactly: "Insufficient data to verify from the uploaded documents."
- Preserve important technical terms exactly as they appear in the paper: Naive RAG, Advanced RAG, Modular RAG, retrieval, generation, indexing, evaluation, benchmarks, metrics, hallucination, missing content.
- Cite using [page X] only. Always use the page number shown after each context chunk. Do not invent page numbers.
- Do NOT include filenames (like "filename.pdf") inside the answer. Source filenames will be shown separately.
- Do NOT include "Context X:" or "[Context X" anywhere in your answer.
- Never make up citations or use [page 0].
- Keep answers focused and concise. Use bullet points only when helpful.

Answer in English."""

# === Arabic Prompt Template ===
ARABIC_SYSTEM_PROMPT = """أنت مساعد مفيد تجيب على الأسئلة بناءً على السياق المقدم فقط.

القواعد:
- ابدأ بالإجابة مباشرة. لا تستخدم "يبدو أن" أو "من المحتمل".
- إذا لم يحتوي السياق على الإجابة، أعد النص التالي بالضبط: "Insufficient data to verify from the uploaded documents."
- حافظ على المصطلحات التقنية الإنجليزية كما هي في البحث: Naive RAG، Advanced RAG، Modular RAG، retrieval، generation، indexing، evaluation، benchmarks، metrics، hallucination، missing content. لا تترجم RAG إلى "راغ" أو "راغي".
- استخدم [صفحة X] فقط للاستشهاد. استخدم رقم الصفحة الموجود بعد كل جزء من السياق. لا تخترع أرقام الصفحات.
- لا تدرج أسماء الملفات داخل الإجابة. سيتم عرض أسماء الملفات بشكل منفصل.
- لا تدرج "Context X:" أو "[Context X" في إجابتك أبدًا.
- لا تخترع استشهادات أو تستخدم [صفحة 0].
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

    Args:
        chunks: List of SearchResult or Chunk objects
        include_scores: If True, include relevance scores

    Returns:
        Formatted context string with source annotations
    """
    lines = []
    for i, result in enumerate(chunks, 1):
        chunk = result.chunk if hasattr(result, "chunk") else result
        source = f"[{chunk.source_file}, page {chunk.page_num}]"
        score = (
            f" (score: {result.score:.2f})" if include_scores and hasattr(result, "score") else ""
        )
        lines.append(f"[{i}]{score}: {chunk.text}\n{source}")
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
    # Phase 4: Update prompt format based on LLM requirements
    return f"""{system_prompt}

Context:
{context}

Question: {question}

Answer (with citations):"""
