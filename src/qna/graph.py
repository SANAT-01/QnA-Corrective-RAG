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

# How many times we're willing to rewrite the question and re-retrieve before
# giving up and answering anyway with a low-confidence flag.
MAX_RETRIES = 2

# How many chunks to pull back per retrieval pass.
TOP_K = 4

# Where the embedded (file-based) Chroma store lives. v0.1 only - from v0.4
# onward this is replaced by a real Chroma server reached over HTTP.
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "qna_docs"


# This is the "State" piece of LangGraph. TypedDict is just a type hint for
# your editor/type-checker - at runtime this is a plain dict, nothing magic.
# Every node function below receives the *whole* state dict as input, but
# only returns the keys it actually changed; LangGraph merges that partial
# dict back into the running state for you between node calls.
class GraphState(TypedDict):
    question: str          # mutated by rewrite_query on each retry
    documents: list[Document]  # the chunks retrieved on the most recent pass
    retry_count: int        # how many times we've already rewritten+retried
    relevant: bool          # grade_documents' verdict on the current documents
    confident: bool         # whether the final answer should be trusted
    generation: str         # the final answer text
    trace: list[str]        # human-readable log of which nodes ran, for debugging


def _get_vector_store() -> Chroma:
    """Open a handle to the (already-ingested) local Chroma store.

    This doesn't create new data - ingest.py is what writes to CHROMA_DIR.
    This just opens the same folder for reading via similarity search.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


# --- Nodes ---------------------------------------------------------------
# Each node below is a plain function: (state) -> dict of updates.
# LangGraph calls these one at a time as it walks the graph, feeding the
# current state in and merging whatever dict comes back.


def retrieve(state: GraphState) -> dict:
    """Embed the question and pull the top-k most similar chunks from Chroma.

    This runs every time we enter the loop - including after rewrite_query
    has changed `state["question"]`, which is exactly how the retry loop
    gets a second (and third) chance at finding relevant context.
    """
    vector_store = _get_vector_store()
    documents = vector_store.similarity_search(state["question"], k=TOP_K)
    return {
        "documents": documents,
        # Note: we append to state["trace"], not mutate it, since this
        # function should only ever return new values, never edit state in place.
        "trace": state["trace"] + [f"retrieve: found {len(documents)} chunk(s)"],
    }


def grade_documents(state: GraphState) -> dict:
    """Ask the LLM whether the retrieved chunks are actually relevant to the question.

    This is the "corrective" part of Corrective RAG: instead of blindly
    generating an answer from whatever got retrieved, we have the LLM judge
    the retrieval quality first. The verdict (`relevant`) is what the
    conditional edge (`decide_next`, below) uses to pick the next node.
    """
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
    """Ask the LLM to rephrase the question for a better retrieval pass.

    Only reached when grade_documents said the context wasn't relevant and
    we still have retries left (see decide_next). Overwriting `question`
    here is what feeds a better-phrased query back into the next `retrieve`.
    """
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
    """Answer the question using the retrieved chunks, citing sources.

    This is the terminal node on every path through the graph - reached
    either because the context was judged relevant, or because we ran out
    of retries and are answering anyway with a low-confidence disclaimer.
    """
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


# --- Conditional edge ------------------------------------------------------
# A normal edge (graph.add_edge) always goes to the same next node. This
# function is a *conditional* edge instead: LangGraph calls it with the
# current state after "grade_documents" runs, and whatever string it
# returns tells LangGraph which node to go to next. This is the only place
# in the whole graph where branching/looping decisions actually get made.
def decide_next(state: GraphState) -> str:
    if state["relevant"]:
        # Good context - skip straight to answering.
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        # Bad context, but we still have retries - rephrase and try again.
        return "rewrite_query"
    # Bad context and out of retries - answer anyway, but generate() will
    # see state["relevant"] is False and mark the answer low-confidence.
    return "generate"


def build_graph():
    # StateGraph(GraphState) tells LangGraph the shape of the state dict
    # that will flow through every node below.
    graph = StateGraph(GraphState)

    # Register each node function under a name - these names are what
    # add_edge/add_conditional_edges below use to wire them together.
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    # Where the graph starts on every invocation.
    graph.set_entry_point("retrieve")

    # Plain edge: retrieve always flows into grade_documents next.
    graph.add_edge("retrieve", "grade_documents")

    # The retry loop: after grading, call decide_next(state) to pick the
    # next node. The dict maps decide_next's possible return values to the
    # actual node names to jump to (here they happen to match 1:1).
    graph.add_conditional_edges(
        "grade_documents",
        decide_next,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )

    # This edge is what closes the loop: after rewriting the query, go
    # back to retrieve (with the new question) rather than straight to
    # grading - we need fresh documents for the new phrasing first.
    graph.add_edge("rewrite_query", "retrieve")

    # generate is always the end of the line.
    graph.add_edge("generate", END)

    # .compile() turns the graph definition into something invokable:
    # app.invoke(initial_state) walks the graph and returns the final state.
    return graph.compile()


# Built once at import time so both main.py and (later) api.py can share
# the same compiled graph instance.
app = build_graph()
