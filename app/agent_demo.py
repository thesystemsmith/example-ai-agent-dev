from app.agent.service import LearningAgent


THREAD_ID = "local-learning-session"


def main() -> None:
    print("Local Stateful Learning Agent")
    print("Type 'exit' to stop.")

    try:
        # KEEP: Build the index and agent only once.
        agent = LearningAgent.from_directory(
            "documents"
        )
    except Exception as error:
        print(f"\nAgent startup failed: {error}")
        return

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        if not question:
            print("Please provide a question.")
            continue

        try:
            result = agent.ask(
                question,
                thread_id=THREAD_ID,
            )

            print(f"\nAgent:\n{result.answer}")

            if result.tools_used:
                print(
                    "\nTools used: "
                    + ", ".join(result.tools_used)
                )
            else:
                print("\nTools used: none")

        except Exception as error:
            print(f"\nAgent request failed: {error}")


if __name__ == "__main__":
    main()