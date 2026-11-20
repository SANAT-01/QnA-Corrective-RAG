"""CLI: ask a question against the ingested documents.

Usage:
    uv run python -m qna.main "your question here"
"""

import sys

from qna.graph import app


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python -m qna.main "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    # Every field from GraphState (qna/graph.py) needs a starting value -
    # LangGraph doesn't default missing keys for you.
    initial_state = {
        "question": question,
        "documents": [],
        "retry_count": 0,
        "relevant": False,
        "confident": False,
        "generation": "",
        "trace": [],
    }

    # Walks the whole compiled graph (retrieve -> grade -> maybe loop ->
    # generate) and returns the final state dict once it hits END.
    result = app.invoke(initial_state)

    print("\n--- Trace ---")
    for step in result["trace"]:
        print(f"  {step}")

    print("\n--- Answer ---")
    print(result["generation"])
    print(f"\n(confident={result['confident']}, retries={result['retry_count']})")


if __name__ == "__main__":
    main()
