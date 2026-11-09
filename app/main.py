"""API FastAPI de tradução inglês -> francês com Helsinki-NLP/opus-mt-en-fr (HuggingFace).

Execução:  uvicorn app.main:app --reload
Docs:      http://127.0.0.1:8000/docs
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from app.config import get_settings
from app.schemas import HealthResponse, TranslationRequest, TranslationResponse
from app.translator import HFTranslator, TranslatorProtocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo uma única vez na inicialização da API."""
    s = get_settings()
    app.state.translator = HFTranslator(s.hf_model, s.device, s.max_chunk_tokens, s.batch_size)
    yield


app = FastAPI(
    title="EN -> FR Translation API",
    description="Traduz textos do inglês para o francês com o modelo Helsinki-NLP/opus-mt-en-fr.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_translator(request: Request) -> TranslatorProtocol:
    """Dependência do FastAPI: entrega à rota o tradutor carregado no startup (ou HTTP 503 se ainda não carregou)."""
    translator = getattr(request.app.state, "translator", None)
    if translator is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo ainda não carregado.")
    return translator


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health(request: Request) -> HealthResponse:
    """Rota de verificação (GET /health): informa se a API está no ar e se o modelo foi carregado. Útil para
    monitoramento.
    """
    return HealthResponse(
        status="ok",
        model=get_settings().hf_model,
        model_loaded=getattr(request.app.state, "translator", None) is not None,
    )


@app.post("/translate", response_model=TranslationResponse, tags=["tradução"])
def translate(
    payload: TranslationRequest,
    translator: TranslatorProtocol = Depends(get_translator),
) -> TranslationResponse:
    """Rota POST /translate: recebe o texto já validado pelo Pydantic, traduz, mede o tempo e devolve JSON. Erros
    inesperados do modelo viram HTTP 500.
    """
    start = time.perf_counter()
    try:
        result = translator.translate(payload.text)
    except Exception as exc:
        logger.exception("Falha na tradução")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Erro ao traduzir o texto.") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Tradução de %d trecho(s) em %.0f ms", result.chunks, elapsed_ms)
    return TranslationResponse(
        original_text=payload.text,
        translated_text=result.text,
        model=translator.model_name,
        chunks=result.chunks,
        elapsed_ms=round(elapsed_ms, 2),
    )
