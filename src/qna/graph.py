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

import json
import logging
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from qna import logging_config  # noqa: F401 - configures logging on import
from qna.config import settings
from qna.exceptions import GenerationError, LLMError
from qna.llm import get_chat_model, get_vector_store

logger = logging.getLogger(__name__)


def source_label(doc: Document) -> str:
    """Human-readable citation for a chunk: filename plus page (PDF) or
    heading (markdown) when available. Shared by grade_documents' trace
    output and api.py's `sources` field, so both describe chunks the same
    way.
    """
    source = doc.metadata.get("source", "unknown")
    if "page" in doc.metadata:
        return f"{source} p.{doc.metadata['page']}"
    if "heading" in doc.metadata:
        return f"{source} #{doc.metadata['heading']}"
    return source


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
    vector_store = get_vector_store()
    documents = vector_store.similarity_search(state["question"], k=settings.top_k)
    logger.info("retrieve: found %d chunk(s)", len(documents))
    return {
        "documents": documents,
        # Note: we append to state["trace"], not mutate it, since this
        # function should only ever return new values, never edit state in place.
        "trace": state["trace"] + [f"retrieve: found {len(documents)} chunk(s)"],
    }


def grade_documents(state: GraphState) -> dict:
    """Ask the LLM to grade each retrieved chunk individually for relevance.

    This is the "corrective" part of Corrective RAG: instead of blindly
    generating an answer from whatever got retrieved, we have the LLM judge
    each chunk's relevance first, with its reasoning. Chunks graded
    irrelevant are dropped before generate() ever sees them; the overall
    `relevant` flag (true if at least one chunk survived) is what the
    conditional edge (`decide_next`, below) uses to pick the next node.
    """
    documents = state["documents"]
    numbered_chunks = "\n\n---\n\n".join(
        f"[{i}] ({source_label(doc)})\n{doc.page_content}" for i, doc in enumerate(documents)
    )
    prompt = (
        "You are grading whether each retrieved chunk below is relevant to a question.\n"
        f"Question: {state['question']}\n\n"
        f"Chunks:\n{numbered_chunks}\n\n"
        "Respond with ONLY a JSON array, one object per chunk, in exactly this form "
        '(no other text before or after it):\n'
        '[{"index": 0, "relevant": true, "reasoning": "one short sentence"}, ...]'
    )
    try:
        response = get_chat_model().invoke(prompt)
    except LLMError:
        raise
    except Exception as e:
        raise GenerationError(f"grade_documents: chat model call failed: {e}") from e

    verdicts = _parse_grading_response(response.content, len(documents))
    if verdicts is None:
        # Fail open: trust the retrieval rather than block the whole answer
        # on a parsing hiccup in the grader's structured output.
        logger.warning("grade_documents: could not parse grading output, keeping all chunks")
        return {
            "documents": documents,
            "relevant": True,
            "trace": state["trace"] + ["grade_documents: parse failed, keeping all chunks (fail-open)"],
        }

    relevant_docs = []
    trace_lines = []
    for verdict in verdicts:
        index = verdict["index"]
        doc = documents[index]
        trace_lines.append(
            f"grade_documents: chunk {index} ({source_label(doc)}) "
            f"relevant={verdict['relevant']}: \"{verdict['reasoning']}\""
        )
        if verdict["relevant"]:
            relevant_docs.append(doc)

    relevant = len(relevant_docs) > 0
    logger.info("grade_documents: kept %d/%d chunk(s), relevant=%s", len(relevant_docs), len(documents), relevant)
    return {
        # If nothing survived grading, keep the original chunks around so a
        # low-confidence generate() still has *something* to work with,
        # rather than an empty context.
        "documents": relevant_docs if relevant else documents,
        "relevant": relevant,
        "trace": state["trace"] + trace_lines,
    }


def _parse_grading_response(raw: str, expected_count: int) -> list[dict] | None:
    """Parse grade_documents' expected JSON array response. Returns None
    (triggering the fail-open path above) if the response isn't usable -
    not just invalid JSON, but also a shape that doesn't match what we
    retrieved, which is just as unusable.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return None
        text = text[start : end + 1]

    try:
        verdicts = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(verdicts, list):
        return None

    parsed = []
    for item in verdicts:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        if not isinstance(index, int) or not (0 <= index < expected_count):
            return None
        parsed.append(
            {
                "index": index,
                "relevant": bool(item.get("relevant")),
                "reasoning": str(item.get("reasoning", "")).strip(),
            }
        )
    return parsed


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
    try:
        response = get_chat_model().invoke(prompt)
    except LLMError:
        raise
    except Exception as e:
        raise GenerationError(f"rewrite_query: chat model call failed: {e}") from e
    rewritten = response.content.strip()
    logger.info("rewrite_query: '%s'", rewritten)
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
    try:
        response = get_chat_model().invoke(prompt)
    except LLMError:
        raise
    except Exception as e:
        raise GenerationError(f"generate: chat model call failed: {e}") from e
    answer = response.content.strip()
    if not confident:
        answer = (
            "[Low confidence - retrieved context may not be fully relevant]\n\n"
            + answer
        )
    logger.info("generate: answer produced (confident=%s)", confident)
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
    if state["retry_count"] < settings.max_retries:
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


def _save_graph_diagram(compiled_graph, path: str = "graph.png") -> None:
    """Render the compiled graph's structure to a PNG via LangGraph's
    built-in Mermaid renderer, so the diagram can never drift from the
    actual node/edge wiring above. Calls out to a free hosted Mermaid
    rendering service, so this is best-effort: no network (or any other
    failure) should ever block the app from starting.
    """
    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
    except Exception as e:  # noqa: BLE001 - best-effort, never fatal
        print(f"(skipping graph diagram: {e})")


# Built once at import time so both main.py and (later) api.py can share
# the same compiled graph instance.
app = build_graph()
_save_graph_diagram(app)
