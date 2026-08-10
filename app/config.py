import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )
    model: str = os.getenv(
        "OLLAMA_MODEL",
        "qwen2.5:3b-instruct",
    )
    temperature: float = float(
        os.getenv("MODEL_TEMPERATURE", "0.2")
    )
    context_size: int = int(
        os.getenv("MODEL_CONTEXT_SIZE", "4096")
    )
    embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "embeddinggemma",
    )


settings = Settings()
