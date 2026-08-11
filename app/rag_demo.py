from app.rag.service import RagService


def main() -> None:
    print("Local RAG Learning Coach — LangChain + Ollama")

    question = input(
        "\nAsk a question from your notes: "
    ).strip()

    if not question:
        print("Please provide a question.")
        return

    try:
        rag_service = RagService.from_directory(
            "documents"
        )

        result = rag_service.ask(
            question,
            top_k=3,
        )

        print(f"\nAnswer:\n{result.answer}")

        print("\nSources:")
        for source in result.sources:
            print(f"- {source}")

        # retrieval details provide basic RAG observability.
        print("\n--- Retrieved evidence ---")

        for rank, retrieved in enumerate(
            result.retrieved_chunks,
            start=1,
        ):
            print(
                f"{rank}. "
                f"{retrieved.chunk.source}:"
                f"{retrieved.chunk.chunk_id} "
                f"(similarity={retrieved.score:.4f})"
            )

    except Exception as error:
        print(f"\nRAG request failed: {error}")


if __name__ == "__main__":
    main()