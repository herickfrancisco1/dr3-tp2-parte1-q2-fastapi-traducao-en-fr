"""Configurações da aplicação, lidas de variáveis de ambiente (prefixo APP_) ou do arquivo .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    hf_model: str = "Helsinki-NLP/opus-mt-en-fr"
    device: int = -1  # -1 = CPU; 0, 1, ... = GPU
    max_chunk_tokens: int = 400  # frases maiores que isso são divididas (limite do modelo: 512)
    batch_size: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
