"""LLM prompt chains used by the self-reflective RAG workflow.

Grading logic (relevance / hallucination / answer-quality) and the core
RAG answer chain live here, isolated from the graph control-flow logic
in `graph.py` and `nodes.py`.
"""
from __future__ import annotations

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
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    """Binary score to assess whether an answer addresses a question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


# ---------------------------------------------------------------------------
# LLM (single shared instance, configured via config.py)
# ---------------------------------------------------------------------------
llm = ChatOllama(model=config.llm_model, temperature=config.llm_temperature)

# ---------------------------------------------------------------------------
# Retrieval grader: is a retrieved chunk relevant to the question?
# ---------------------------------------------------------------------------
_retrieval_grader_system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
_retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _retrieval_grader_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = _retrieval_grader_prompt | llm.with_structured_output(GradeDocuments)

# ---------------------------------------------------------------------------
# Core RAG answer chain
# ---------------------------------------------------------------------------
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Hallucination grader: is the generation grounded in retrieved documents?
# ---------------------------------------------------------------------------
_hallucination_grader_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
_hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _hallucination_grader_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = _hallucination_grader_prompt | llm.with_structured_output(GradeHallucinations)

# ---------------------------------------------------------------------------
# Answer grader: does the generation actually answer the question?
# ---------------------------------------------------------------------------
_answer_grader_system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
_answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _answer_grader_system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = _answer_grader_prompt | llm.with_structured_output(GradeAnswer)

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
