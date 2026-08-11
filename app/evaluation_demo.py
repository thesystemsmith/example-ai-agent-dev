from dataclasses import dataclass

from app.agent.service import LearningAgent


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    question: str
    thread_id: str
    expected_tools: tuple[str, ...]


# KEEP: Evaluate agent decisions instead of unpredictable answer wording.
CASES = (
    EvaluationCase(
        name="Greeting avoids retrieval",
        question="Hello!",
        thread_id="eval-greeting",
        expected_tools=(),
    ),
    EvaluationCase(
        name="Learning request uses retrieval",
        question="Teach me RAG",
        thread_id="eval-learning",
        expected_tools=("search_notes",),
    ),
    EvaluationCase(
        name="Quiz uses retrieval",
        question="Quiz me about AI agents",
        thread_id="eval-memory",
        expected_tools=("search_notes",),
    ),
    EvaluationCase(
        name="Quiz answer uses memory",
        question=(
            "An agent combines a model with tools, instructions, "
            "state, and a control loop."
        ),
        # Reusing the ID restores the previous quiz conversation.
        thread_id="eval-memory",
        expected_tools=(),
    ),
)


def main() -> None:
    agent = LearningAgent.from_directory("documents")
    passed_count = 0

    for case in CASES:
        result = agent.ask(
            question=case.question,
            thread_id=case.thread_id,
        )

        passed = (
            result.tools_used == case.expected_tools
            and bool(result.answer.strip())
        )

        passed_count += int(passed)

        print(f"\n{'PASS' if passed else 'FAIL'}: {case.name}")
        print(f"Expected tools: {case.expected_tools}")
        print(f"Actual tools:   {result.tools_used}")
        print(f"Answer: {result.answer}")

    print(f"\nResult: {passed_count}/{len(CASES)} passed")


if __name__ == "__main__":
    main()