"""Camada de serviço: tradução EN -> FR com o pipeline `translation` do HuggingFace."""

import logging
from dataclasses import dataclass
from typing import Protocol

from app.text_chunking import chunk_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationResult:
    text: str
    chunks: int


class TranslatorProtocol(Protocol):
    model_name: str

    def translate(self, text: str) -> TranslationResult: ...


class HFTranslator:
    """Tradutor baseado em um modelo MarianMT (Helsinki-NLP/opus-mt-en-fr)."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-fr",
        device: int = -1,
        max_chunk_tokens: int = 400,
        batch_size: int = 8,
    ) -> None:
        from transformers import pipeline

        logger.info("Carregando modelo '%s' (device=%s)...", model_name, device)
        self.model_name = model_name
        self._pipe = pipeline("translation", model=model_name, device=device)
        self._max_chunk_tokens = max_chunk_tokens
        self._batch_size = batch_size
        logger.info("Modelo '%s' carregado.", model_name)

    def _count_tokens(self, text: str) -> int:
        return len(self._pipe.tokenizer.encode(text))

    def translate(self, text: str) -> TranslationResult:
        paragraphs = chunk_text(text, self._count_tokens, self._max_chunk_tokens)
        flat_chunks = [chunk for paragraph in paragraphs for chunk in paragraph]

        # Todos os trechos são traduzidos em lote (batch), aproveitando melhor CPU/GPU.
        outputs = self._pipe(flat_chunks, batch_size=self._batch_size, max_length=512)
        translations = iter(o["translation_text"] for o in outputs)

        # Remonta o texto preservando a divisão em parágrafos.
        translated_paragraphs = [" ".join(next(translations) for _ in p) for p in paragraphs]
        return TranslationResult(text="\n\n".join(translated_paragraphs), chunks=len(flat_chunks))
