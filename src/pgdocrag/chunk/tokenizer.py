"""Token counting using the embedding model's own vocabulary.

The chunk size limit only means something if it is measured with the same
tokenizer the model uses. Counting words or characters instead would let chunks
silently exceed the model's 512-token window, where the overflow is discarded
without any error.
"""

from __future__ import annotations

from .. import config

_CHARS_PER_TOKEN_ESTIMATE = 4


class TokenCounter:
    """Wraps the model tokenizer, falling back to an estimate if unavailable.

    The fallback exists so the pipeline still runs offline; it is deliberately
    conservative because undercounting causes silent truncation at embed time.
    """

    def __init__(self, model_name: str = config.EMBED_MODEL_NAME) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._loaded = False
        self.exact = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_pretrained(self.model_name)
            self.exact = True
        except Exception:
            self._tokenizer = None
            self.exact = False

    def count(self, text: str) -> int:
        if not text:
            return 0
        self._load()
        if self._tokenizer is None:
            return max(1, -(-len(text) // _CHARS_PER_TOKEN_ESTIMATE))
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def describe(self) -> str:
        self._load()
        return (
            f"exact ({self.model_name})"
            if self.exact
            else f"estimated (~{_CHARS_PER_TOKEN_ESTIMATE} chars/token)"
        )


_counter: TokenCounter | None = None


def get_counter() -> TokenCounter:
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def count_tokens(text: str) -> int:
    return get_counter().count(text)
