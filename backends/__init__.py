from enum import Enum

from .base import Backend


class BackendType(str, Enum):
    ollama = "ollama"
    openai = "openai"


def get_backend(backend_type: BackendType) -> Backend:
    """Factory function to create backend instances."""
    match backend_type:
        case BackendType.ollama:
            from .ollama import OllamaBackend

            return OllamaBackend()
        case BackendType.openai:
            from .openai import OpenAIBackend

            return OpenAIBackend()


__all__ = [
    "Backend",
    "BackendType",
    "get_backend",
]
