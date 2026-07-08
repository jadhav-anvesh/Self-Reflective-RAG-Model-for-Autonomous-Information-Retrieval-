"""LLM prompt chains used by the self-reflective RAG workflow.

Grading logic (relevance / hallucination / answer-quality) and the core
RAG answer chain live here, isolated from the graph control-flow logic
in `graph.py` and `nodes.py`.
"""
from __future__ import annotations

from typing import Literal

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config import config
from src.workflow.query_cleaning import clean_rewritten_question


# ---------------------------------------------------------------------------
# Structured grading schemas
# ---------------------------------------------------------------------------
# IMPORTANT: `method="json_schema"` is explicit and load-bearing below, not
# stylistic. The pinned `langchain-ollama>=0.2.0,<0.3` defaults
# `with_structured_output()` to `method="function_calling"`, which binds the
# schema as an Ollama *tool* and parses the result with
# `PydanticToolsParser(first_tool_only=True)`. That parser's `parse_result()`
# does `return tool_calls[0] if tool_calls else None` -- if llama3.2 responds
# with plain text instead of actually invoking the bound tool (which local
# models do intermittently, more so as context length/complexity grows), the
# result is `None` with NO exception raised. That's the exact, verified
# source of `AttributeError: 'NoneType' object has no attribute 'binary_score'`.
# `method="json_schema"` instead uses Ollama's grammar-constrained `format`
# parameter + `PydanticOutputParser`, which *raises* `OutputParserException`
# on any failure to produce valid, schema-conforming output -- turning this
# failure mode from a silent `None` into a loud, catchable exception.
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(description="Documents are relevant to the question")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: Literal["yes", "no"] = Field(description="Answer is grounded in the facts")


class GradeAnswer(BaseModel):
    """Binary score to assess whether an answer addresses a question."""

    binary_score: Literal["yes", "no"] = Field(description="Answer addresses the question")


# ---------------------------------------------------------------------------
# LLM (single shared instance, configured via config.py)
# ---------------------------------------------------------------------------
# `request_timeout` is NOT a real ChatOllama field -- it was silently ignored
# (model_config = {'extra': 'ignore', ...}), so no client-side timeout was
# ever actually applied. `client_kwargs={"timeout": ...}` is the real,
# verified field name (passed through to httpx). `num_predict` caps worst-case
# generation length as defense-in-depth on top of the enum-constrained
# grading schemas above -- neither alone is a substitute for the other.
llm = ChatOllama(
    model=config.llm_model,
    temperature=config.llm_temperature,
    client_kwargs={"timeout": 60},
)

# ---------------------------------------------------------------------------
# Retrieval grader: is a retrieved chunk relevant to the question?
# ---------------------------------------------------------------------------
_retrieval_grader_system = """You are evaluating whether a retrieved document should be kept for answer generation.

Return "yes" if the document contains information that directly answers the question OR provides evidence needed to answer part of the question.

Return "no" if the document:
- is only loosely related,
- mentions similar keywords without useful information,
- contains only generic background,
- or would not improve the final answer.

A document does NOT need to answer the entire question.
If it contributes useful evidence, return "yes".

Respond only with:
yes
or
no"""
_retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _retrieval_grader_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = _retrieval_grader_prompt | llm.with_structured_output(GradeDocuments, method="json_schema")

# ---------------------------------------------------------------------------
# Core RAG answer chain
# ---------------------------------------------------------------------------
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Hallucination grader: is the generation grounded in retrieved documents?
# ---------------------------------------------------------------------------
_hallucination_grader_system = """You are verifying whether an answer is fully supported by the retrieved documents.

Do NOT penalize answers for being shorter than the retrieved documents if every claim made is supported.
Return "yes" ONLY if:

- every factual statement in the answer is supported by the retrieved documents,
- no claim requires outside knowledge,
- no entities, numbers, definitions or examples are invented,
- no unsupported assumptions are introduced.

Return "no" if:

- even one factual claim lacks evidence,
- the answer contradicts the retrieved documents,
- the answer includes additional information not found in the retrieved documents,
- the answer speculates or generalizes beyond the provided evidence.

Minor wording differences and paraphrasing are acceptable.

If you are uncertain, return "no".

Respond with only:

yes

or

no"""
_hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _hallucination_grader_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = _hallucination_grader_prompt | llm.with_structured_output(
    GradeHallucinations, method="json_schema"
)

# ---------------------------------------------------------------------------
# Answer grader: does the generation actually answer the question?
# ---------------------------------------------------------------------------
_answer_grader_system = """You are evaluating whether the generated answer sufficiently answers the user's question.

Return "yes" only if:

- the answer addresses every important part of the user's question,
- all requested information is present,
- the answer is supported by the retrieved documents,
- the answer is specific rather than generic.

Return "no" if:

- any important part of the question is unanswered,
- important information is missing,
- the answer is vague or overly generic,
- the answer avoids the question.

Do NOT require perfect wording.

Do NOT reject an answer simply because it is concise.

If unsure, return "no".

Respond only with:

yes

or

no"""
_answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _answer_grader_system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = _answer_grader_prompt | llm.with_structured_output(GradeAnswer, method="json_schema")

# ---------------------------------------------------------------------------
# Question rewriter: used when retrieved documents are all irrelevant
# ---------------------------------------------------------------------------
_question_rewriter_system = """You are a question re-writer that converts an input question into a
better version optimized for vector-store retrieval, by reasoning about
its underlying semantic intent.

Output rules (strict):
- Output ONLY the rewritten question -- nothing else.
- Do not include labels such as "Improved Question:", "Rewritten Question:", or "Question:".
- Do not include any explanation, reasoning, or a "Reasoning:"/"Explanation:" section.
- Output must be a single line containing only the rewritten question, with no surrounding quotes."""
_question_rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _question_rewriter_system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Respond with only the rewritten question.",
        ),
    ]
)
# clean_rewritten_question() is defense-in-depth: the prompt above asks the
# model to output only the question, but small/local models don't always
# comply (observed emitting "Improved Question: ...\n\nReasoning: ..."), so
# this strips that pattern rather than trusting the model unconditionally.
# The retriever must receive only the rewritten question text.
question_rewriter = _question_rewriter_prompt | llm | StrOutputParser() | RunnableLambda(clean_rewritten_question)
