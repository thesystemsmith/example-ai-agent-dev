SYSTEM_PROMPT = """
You are a patient AI learning coach.

Explain technical concepts:
- accurately,
- simply,
- practically,
- with a clear example,
- without unnecessary jargon.
"""


def create_explanation_messages(topic: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": (
                f"Teach me this topic: {topic}. "
                "Keep the explanation concise and practical."
            ),
        },
    ]
