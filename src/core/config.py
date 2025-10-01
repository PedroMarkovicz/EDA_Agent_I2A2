"""
Centralizador de configurações e variáveis globais do sistema.
Gerencia configurações de ambiente, parâmetros de LLMs, limites de execução
e outras constantes utilizadas em todo o sistema EDA.
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    """Configuração centralizada do sistema EDA."""

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="Chave da API do OpenAI")

    # LangChain Configuration
    langchain_tracing_v2: bool = Field(False, description="Habilitar tracing do LangChain")
    langchain_api_key: Optional[str] = Field(None, description="Chave da API do LangChain")

    # Application Configuration
    app_env: str = Field("development", description="Ambiente da aplicação")
    log_level: str = Field("INFO", description="Nível de log")

    # File Upload Settings
    max_upload_size_mb: int = Field(200, description="Tamanho máximo de upload em MB")
    allowed_extensions: str = Field("csv", description="Extensões permitidas")

    # Code Execution Settings
    execution_timeout: int = Field(30, description="Timeout de execução em segundos")
    safe_mode: bool = Field(True, description="Modo seguro de execução")

    # LLM Model Configuration
    llm_model: str = Field("gpt-5-nano-2025-08-07", description="Modelo LLM a ser usado")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("log_level")
    def validate_log_level(cls, v):
        """Valida se o nível de log é válido."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL deve ser um de: {valid_levels}")
        return v.upper()

    @validator("app_env")
    def validate_app_env(cls, v):
        """Valida o ambiente da aplicação."""
        valid_envs = ["development", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"APP_ENV deve ser um de: {valid_envs}")
        return v.lower()

    @validator("max_upload_size_mb")
    def validate_upload_size(cls, v):
        """Valida tamanho máximo de upload."""
        if v <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_MB deve ser maior que 0")
        return v

    def get_allowed_extensions(self) -> list[str]:
        """Retorna lista de extensões permitidas."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    def is_development(self) -> bool:
        """Verifica se está em ambiente de desenvolvimento."""
        return self.app_env == "development"

    def is_production(self) -> bool:
        """Verifica se está em ambiente de produção."""
        return self.app_env == "production"


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Obtém a instância singleton da configuração."""
    global _config
    if _config is None:
        # Carregar variáveis do .env
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
        _config = Config(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "200")),
            allowed_extensions=os.getenv("ALLOWED_EXTENSIONS", "csv"),
            execution_timeout=int(os.getenv("EXECUTION_TIMEOUT", "30")),
            safe_mode=os.getenv("SAFE_MODE", "true").lower() == "true"
        )
    return _config


def reload_config() -> Config:
    """Recarrega a configuração (útil para testes)."""
    global _config
    _config = None
    return get_config()