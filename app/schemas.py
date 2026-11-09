"""Modelos Pydantic de entrada e saída da API."""

from pydantic import BaseModel, Field, field_validator


class TranslationRequest(BaseModel):
    """Corpo da requisição POST /translate: o texto em inglês a ser traduzido."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Texto em inglês (frase curta ou texto longo com vários parágrafos).",
        examples=["Hello! How are you today?"],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Rejeita textos só com espaços (o min_length=1 sozinho aceitaria '   '). O erro vira HTTP 422."""
        if not value.strip():
            raise ValueError("o texto não pode conter apenas espaços")
        return value


class TranslationResponse(BaseModel):
    """Corpo da resposta de /translate: idiomas, texto original, tradução, modelo usado e tempo gasto."""

    source_language: str = "en"
    target_language: str = "fr"
    original_text: str
    translated_text: str
    model: str
    chunks: int = Field(..., description="Quantos trechos foram traduzidos (textos longos são divididos).")
    elapsed_ms: float


class HealthResponse(BaseModel):
    """Corpo da resposta de GET /health."""

    status: str
    model: str
    model_loaded: bool
