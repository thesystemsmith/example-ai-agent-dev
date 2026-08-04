from dataclasses import dataclass

from ollama import Client, ResponseError

from app.config import settings
from app.prompts import create_explanation_messages
from app.schemas import LearningExplanation


@dataclass
class ModelMetrics:
    prompt_tokens: int
    output_tokens: int
    total_duration_ms: float


@dataclass
class ExplanationResult:
    explanation: LearningExplanation
    metrics: ModelMetrics


client = Client(host=settings.ollama_host)


def explain_topic(topic: str) -> ExplanationResult:
    messages = create_explanation_messages(topic)

    try:
        response = client.chat(
            model=settings.model,
            messages=messages,
            format=LearningExplanation.model_json_schema(),
            stream=False,
            options={
                "temperature": settings.temperature,
                "num_ctx": settings.context_size,
            },
        )
    except ResponseError as error:
        raise RuntimeError(
            f"Ollama request failed: {error.error}"
        ) from error
    except ConnectionError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Ensure Ollama is running."
        ) from error

    explanation = LearningExplanation.model_validate_json(
        response.message.content
    )

    metrics = ModelMetrics(
        prompt_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
        total_duration_ms=(response.total_duration or 0) / 1_000_000,
    )

    return ExplanationResult(
        explanation=explanation,
        metrics=metrics,
    )
