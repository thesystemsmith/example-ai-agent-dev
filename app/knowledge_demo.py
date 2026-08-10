from app.knowledge.retriever import KnowledgeRetriever


def main() -> None:
    print("Local Knowledge Search — Ollama + FAISS")

    question = input("\nSearch your notes: ").strip()

    if not question:
        print("Please provide a question.")
        return

    try:
        retriever = KnowledgeRetriever.from_directory("documents")
        results = retriever.search(question, top_k=3)

        print(f"\nIndexed chunks: {retriever.chunk_count}")

        for rank, result in enumerate(results, start=1):
            print("\n" + "-" * 60)
            print(f"Result: {rank}")
            print(f"Source: {result.chunk.source}")
            print(f"Chunk ID: {result.chunk.chunk_id}")
            print(f"Similarity: {result.score:.4f}")
            print(result.chunk.content)

    except Exception as error:
        print(f"\nKnowledge search failed: {error}")


if __name__ == "__main__":
    main()