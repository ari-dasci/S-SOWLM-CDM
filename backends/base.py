from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        """Return the assistant reply.

        `max_tokens` caps the generated reply. Each provider spells it
        differently, so implementations translate it rather than pass it on.

        `think` turns a reasoning-capable model's internal trace on or off;
        `None` leaves the provider default alone. It belongs in the backend
        contract rather than in a provider-specific option because it changes
        what is being measured: with the trace on, a single sentiment
        classification costs 6-18 s instead of 0.2-0.4 s, and the reply is the
        end of a chain of reasoning rather than a direct judgement. Providers
        without the notion ignore it.
        """
        ...

    @abstractmethod
    def pull(self, model: str) -> None: ...
