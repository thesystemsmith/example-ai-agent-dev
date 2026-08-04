from pydantic import BaseModel, Field


class LearningExplanation(BaseModel):
    topic: str = Field(
        description="The main topic being explained."
    )

    summary: str = Field(
        description="A short beginner-friendly explanation."
    )

    key_points: list[str] = Field(
        description="Three to five important points."
    )

    example: str = Field(
        description="A practical example."
    )

    check_question: str = Field(
        description="One question to test understanding."
    )
