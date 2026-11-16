from typing import final, override

from .base import Backend


@final
class OpenAIBackend(Backend):
    """Backend implementation using OpenAI API.

    This backend connects to OpenAI's API for chat completions.

    Example:
    ```python
        >>> backend = OpenAIBackend()
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> response = backend.chat("gpt-4o-mini", messages)
        >>> print(response)
        'Hello! How can I assist you today?'
    ```

    Note:
        Requires the OPENAI_API_KEY environment variable to be set.
    """

    def __init__(self, api_key: str | None = None):
        """Initialize OpenAI backend.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    @override
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        """Send a chat completion request to OpenAI.

        `think` is accepted for the shared backend signature and ignored: the
        hosted models expose no equivalent switch.
        """
        extra = {}
        if max_tokens is not None:
            # The Responses API spells it max_output_tokens.
            extra["max_output_tokens"] = max_tokens

        response = self._client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            **extra,
        )
        return response.output_text or ""

    @override
    def pull(self, model: str) -> None:
        """No-op for OpenAI - models are accessed via API."""
