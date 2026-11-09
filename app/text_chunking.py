"""Segmentação de textos para tradução com modelos MarianMT (Helsinki-NLP/opus-mt-*).

Por que segmentar?
- O modelo aceita no máximo 512 tokens por entrada (acima disso o texto seria truncado).
- Os modelos OPUS-MT foram treinados com pares de FRASES. Nos testes, um parágrafo de 4 frases
  enviado de uma vez teve a última frase OMITIDA na tradução. Traduzindo frase a frase (em lote),
  nenhuma frase se perde.

Estratégia: texto -> parágrafos -> frases (cada frase é um item do lote). Uma frase que sozinha
excede `max_tokens` é quebrada por palavras.
"""

import re
from collections.abc import Callable

TokenCounter = Callable[[str], int]

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


def split_paragraphs(text: str) -> list[str]:
    """Separa o texto em parágrafos (blocos divididos por linha em branco), descartando os vazios."""
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """Separa um parágrafo em frases, quebrando após '.', '!', '?' ou ';' seguidos de espaço."""
    return [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]


def _split_long_sentence(sentence: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Quebra por palavras uma frase que, sozinha, excede o limite de tokens."""
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_paragraph(paragraph: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Uma frase por trecho; frases longas demais são divididas em pedaços de até `max_tokens`."""
    chunks: list[str] = []
    for sentence in split_sentences(paragraph):
        if count_tokens(sentence) > max_tokens:
            chunks.extend(_split_long_sentence(sentence, count_tokens, max_tokens))
        else:
            chunks.append(sentence)
    return chunks


def chunk_text(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[list[str]]:
    """Devolve uma lista de parágrafos, cada um como lista de trechos traduzíveis."""
    return [chunk_paragraph(p, count_tokens, max_tokens) for p in split_paragraphs(text)]
