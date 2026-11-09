"""Testes da segmentação de texto (funções puras: rodam sem o modelo)."""

from app.text_chunking import chunk_paragraph, chunk_text, split_paragraphs


def count_words(text: str) -> int:
    """Contador simples (1 palavra = 1 token) para testar a lógica sem o tokenizer real."""
    return len(text.split())


def test_short_sentence_is_a_single_chunk():
    assert chunk_text("Hello world.", count_words, 50) == [["Hello world."]]


def test_each_sentence_becomes_one_chunk():
    paragraph = "One two three. Four five six? Seven eight nine!"
    assert chunk_paragraph(paragraph, count_words, 50) == ["One two three.", "Four five six?", "Seven eight nine!"]


def test_very_long_sentence_is_split_by_words():
    sentence = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_paragraph(sentence, count_words, 4)
    assert chunks == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    assert all(count_words(c) <= 4 for c in chunks)


def test_paragraphs_are_preserved():
    text = "First paragraph.\n\n  \nSecond paragraph."
    assert split_paragraphs(text) == ["First paragraph.", "Second paragraph."]
    assert chunk_text(text, count_words, 50) == [["First paragraph."], ["Second paragraph."]]
