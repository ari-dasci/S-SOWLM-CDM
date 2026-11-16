from typing import final, override

import ollama

from .base import Backend


@final
class OllamaBackend(Backend):
    """Backend implementation using Ollama for local LLM inference.

    This backend connects to a running Ollama instance to perform chat completions.

    Example:
    ```python
        >>> backend = OllamaBackend()
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> response = backend.chat("ministral-3:3b", messages)
        >>> print(response)
        'Hello! 😊 How can I assist you today? Whether you have questions, need help with something, or just want to chat, I'm here for you! What’s on your mind?'
    ```
    Note:
         Requires Ollama to be running locally. Install and start with: https://ollama.com
    """

    def __init__(self):
        """Initialize Ollama backend"""

        self._client = ollama

    @override
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        think: bool | None = None,
        draft_num_predict: int | None = None,
    ) -> str:
        """Send a chat completion request to Ollama"""
        options: dict[str, float | int] = {"temperature": temperature}
        if draft_num_predict is not None:
            # 0 disables a model's speculative-decoding draft.
            options["draft_num_predict"] = draft_num_predict
        if max_tokens is not None:
            # Ollama caps the reply with num_predict; there is no max_tokens.
            options["num_predict"] = max_tokens

        # Omitted rather than passed as None: models without the capability
        # reject the parameter, so sending it unconditionally would break every
        # non-reasoning model.
        extra = {} if think is None else {"think": think}

        response = self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            messages=messages,
            tools=None,
            options=options,
            **extra,
        )

        # `.thinking` carries the reasoning trace and `.content` the answer, so
        # the parser downstream sees a bare label either way. Only the
        # answer is returned; the trace is not part of the recorded output.
        return str(response.message.content)

    @override
    def pull(self, model: str) -> None:
        """Download `model` only if it is not installed.

        Never re-pulls an installed tag: a pull resolves the tag again and can
        silently replace the weights a collection was started with (the
        September 2026 `gemma4:e4b` update added a draft model, and a re-pull
        of `qwen3.6:35b` started fetching new layers). Updating a model is a
        deliberate act, done by hand with `ollama pull`, and recorded.
        """

        installed = {m.model for m in ollama.list().models}
        if model in installed:
            return

        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = None
            current_status = ""

            for update in ollama.pull(model, stream=True):
                status = update.get("status", "")
                completed = update.get("completed")
                total = update.get("total")

                if total and completed:
                    if task is None or status != current_status:
                        if task is not None:
                            progress.remove_task(task)
                        task = progress.add_task(status, total=total)
                        current_status = status
                    progress.update(task, completed=completed)
                else:
                    if task is not None:
                        progress.remove_task(task)
                        task = None
                    progress.console.print(f"[cyan]{status}")
