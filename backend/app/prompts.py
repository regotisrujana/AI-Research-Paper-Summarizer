from langchain_core.prompts import PromptTemplate

BASE_RULES = """Use only the uploaded research paper context.
If the context does not answer the question at all, say exactly:
Information not found in uploaded research papers.
If only some requested sections are supported, answer only those supported sections and omit unsupported headings.
Never write empty headings, placeholder bullets, "No information available", or "Not found in uploaded paper content".
Use simple academic language and keep citations grounded in the provided chunks."""

FACTUAL_PROMPT = PromptTemplate.from_template(
    BASE_RULES
    + """

Context:
{context}

Question:
{question}

Give a direct factual answer."""
)

EXPLANATION_PROMPT = PromptTemplate.from_template(
    BASE_RULES
    + """

Context:
{context}

Question:
{question}

Explain the answer for a beginner without adding outside facts."""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    BASE_RULES
    + """

Context:
{context}

Question:
{question}

Write a structured summary with headings and bullet points.
Omit any heading that is not supported by the retrieved context."""
)

COMPARISON_PROMPT = PromptTemplate.from_template(
    BASE_RULES
    + """

Context:
{context}

Question:
{question}

Compare the requested items using a compact table-style explanation in text."""
)

RESEARCH_GAP_PROMPT = PromptTemplate.from_template(
    BASE_RULES
    + """

Context:
{context}

Question:
{question}

Identify only limitations, unresolved issues, and future research directions found in the paper.
Omit unsupported headings."""
)

VIVA_PROMPT = PromptTemplate.from_template(
    BASE_RULES
    + """

Context:
{context}

Question:
{question}

Generate viva-style questions with short answers using only the paper context."""
)

FOLLOW_UP_PROMPT = PromptTemplate.from_template(
    BASE_RULES
    + """

Prior conversation is available through memory. Resolve pronouns such as it, its, this, and that from the prior paper discussion.

Context:
{context}

Question:
{question}

Answer the follow-up clearly."""
)


def prompt_for(query_type: str) -> PromptTemplate:
    return {
        "summary": SUMMARY_PROMPT,
        "comparison": COMPARISON_PROMPT,
        "research gap": RESEARCH_GAP_PROMPT,
        "viva": VIVA_PROMPT,
        "follow-up": FOLLOW_UP_PROMPT,
        "explanation": EXPLANATION_PROMPT,
    }.get(query_type, FACTUAL_PROMPT)
