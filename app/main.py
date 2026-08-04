from pydantic import ValidationError

from app.llm import explain_topic


def print_explanation(result) -> None:
    explanation = result.explanation

    print(f"\nTopic: {explanation.topic}")
    print(f"\nSummary:\n{explanation.summary}")

    print("\nKey points:")
    for index, point in enumerate(explanation.key_points, start=1):
        print(f"{index}. {point}")

    print(f"\nExample:\n{explanation.example}")
    print(f"\nCheck your understanding:\n{explanation.check_question}")

    print("\n--- Model metrics ---")
    print(f"Prompt tokens: {result.metrics.prompt_tokens}")
    print(f"Output tokens: {result.metrics.output_tokens}")
    print(f"Duration: {result.metrics.total_duration_ms:.2f} ms")


def main() -> None:
    print("Local AI Learning Coach — Ollama")

    topic = input("\nWhat would you like to learn? ").strip()

    if not topic:
        print("Please provide a topic.")
        return

    try:
        result = explain_topic(topic)
        print_explanation(result)
    except ValidationError as error:
        print(f"\nInvalid structured response:\n{error}")
    except RuntimeError as error:
        print(f"\n{error}")
    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
