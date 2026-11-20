"""The Corrective RAG graph.

This file is the core of the whole project. A LangGraph graph is built from
three pieces:

  1. State   - a shared dict-like object (here, a TypedDict called GraphState)
               that flows through every step. Each node reads from it and
               returns a partial update that gets merged back in.
  2. Nodes   - plain functions `(state) -> dict` that do one unit of work:
               retrieve documents, grade them, rewrite the query, or
               generate the final answer.
  3. Edges   - connections between nodes. A normal edge always goes to the
               same next node. A *conditional* edge runs a routing function
               against the state and picks the next node dynamically - this
               is how the "bad context -> retry" loop is implemented below.

The graph itself:

    START -> retrieve -> grade_documents --(relevant)--------------> generate -> END
                  ^              |
                  |         (not relevant, retries < MAX_RETRIES)
                  |              v
                  +------- rewrite_query
                              |
                 (not relevant, retries >= MAX_RETRIES)
                              v
                           generate (low_confidence=True) -> END
"""

from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from qna.llm import get_chat_model, get_embeddings

MAX_RETRIES = 2
TOP_K = 4
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "qna_docs"


class GraphState(TypedDict):
    question: str
    documents: list[Document]
    retry_count: int
    relevant: bool
    confident: bool
    generation: str
    trace: list[str]


def _get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def retrieve(state: GraphState) -> dict:
    """Embed the question and pull the top-k most similar chunks from Chroma."""
    vector_store = _get_vector_store()
    documents = vector_store.similarity_search(state["question"], k=TOP_K)
    return {
        "documents": documents,
        "trace": state["trace"] + [f"retrieve: found {len(documents)} chunk(s)"],
    }


def grade_documents(state: GraphState) -> dict:
    """Ask the LLM whether the retrieved chunks are actually relevant to the question."""
    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
    prompt = (
        "You are grading whether retrieved context is relevant to a question.\n"
        f"Question: {state['question']}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Does this context contain information relevant to answering the question? "
        "Answer with exactly one word: 'yes' or 'no'."
    )
    response = get_chat_model().invoke(prompt)
    verdict = response.content.strip().lower()
    relevant = verdict.startswith("yes")
    return {
        "relevant": relevant,
        "trace": state["trace"] + [f"grade_documents: relevant={relevant}"],
    }


def rewrite_query(state: GraphState) -> dict:
    """Ask the LLM to rephrase the question for a better retrieval pass."""
    prompt = (
        "The following question did not retrieve relevant context from a document "
        "search. Rewrite it to be clearer and more likely to match relevant documents. "
        "Return only the rewritten question, nothing else.\n\n"
        f"Original question: {state['question']}"
    )
    response = get_chat_model().invoke(prompt)
    rewritten = response.content.strip()
    return {
        "question": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": state["trace"] + [f"rewrite_query: '{rewritten}'"],
    }


def generate(state: GraphState) -> dict:
    """Answer the question using the retrieved chunks, citing sources."""
    confident = state["relevant"]
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in state["documents"]
    )
    prompt = (
        "Answer the question using only the context below. Cite the source "
        "filename(s) you used. If the context does not contain the answer, "
        "say so plainly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}"
    )
    response = get_chat_model().invoke(prompt)
    answer = response.content.strip()
    if not confident:
        answer = (
            "[Low confidence - retrieved context may not be fully relevant]\n\n"
            + answer
        )
    return {
        "generation": answer,
        "confident": confident,
        "trace": state["trace"] + ["generate: answer produced"],
    }


def decide_next(state: GraphState) -> str:
    """Conditional edge: route based on relevance and remaining retries."""
    if state["relevant"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"
    return "generate"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_next,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


app = build_graph()
