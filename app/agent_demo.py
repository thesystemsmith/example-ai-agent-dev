from app.agent.service import LearningAgent


def main() -> None:
    print("Local Tool-Using Learning Agent")

    question = input("\nAsk the agent: ").strip()

    if not question:
        print("Please provide a question.")
        return

    try:
        # EXPLAIN: This builds the knowledge index once.
        agent = LearningAgent.from_directory(
            "documents"
        )

        result = agent.ask(question)

        print(f"\nAnswer:\n{result.answer}")

        print("\n--- Agent trace ---")

        if result.tools_used:
            print(
                "Tools used: "
                + ", ".join(result.tools_used)
            )
        else:
            print("Tools used: none")

    except Exception as error:
        print(f"\nAgent request failed: {error}")


if __name__ == "__main__":
    main()