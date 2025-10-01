"""
Sistema de logging centralizado para o EDA Agent.
Configura logs estruturados com diferentes níveis baseados nas
configurações do arquivo .env, incluindo formatação específica
para análise de dados e operações do sistema.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime

from .config import get_config


class EDALogger:
    """Logger centralizado para o sistema EDA."""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str = "eda_agent") -> logging.Logger:
        """Obtém ou cria um logger com configuração padronizada."""
        if name in cls._loggers:
            return cls._loggers[name]

        config = get_config()
        logger = logging.getLogger(name)

        # Evita duplicação de handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Configurar nível baseado no .env
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Formatter com timestamp e contexto
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler para arquivo (apenas em produção)
        if config.is_production():
            cls._add_file_handler(logger, log_level, formatter)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def _add_file_handler(cls, logger: logging.Logger, level: int, formatter: logging.Formatter) -> None:
        """Adiciona handler de arquivo para logs."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"eda_agent_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    @classmethod
    def log_data_operation(cls, logger_name: str, operation: str, details: dict) -> None:
        """Log específico para operações de dados."""
        logger = cls.get_logger(logger_name)
        logger.info(f"DATA_OP: {operation} | {details}")

    @classmethod
    def log_analysis_step(cls, logger_name: str, analysis_type: str, status: str, metadata: Optional[dict] = None) -> None:
        """Log específico para etapas de análise."""
        logger = cls.get_logger(logger_name)
        metadata_str = f" | {metadata}" if metadata else ""
        logger.info(f"ANALYSIS: {analysis_type} - {status}{metadata_str}")

    @classmethod
    def log_llm_call(cls, logger_name: str, model: str, tokens_used: Optional[int] = None, duration: Optional[float] = None) -> None:
        """Log específico para chamadas de LLM."""
        logger = cls.get_logger(logger_name)
        metrics = []
        if tokens_used:
            metrics.append(f"tokens: {tokens_used}")
        if duration:
            metrics.append(f"duration: {duration:.2f}s")
        metrics_str = f" | {', '.join(metrics)}" if metrics else ""
        logger.info(f"LLM_CALL: {model}{metrics_str}")

    @classmethod
    def log_error_with_context(cls, logger_name: str, error: Exception, context: dict) -> None:
        """Log de erro com contexto adicional."""
        logger = cls.get_logger(logger_name)
        logger.error(f"ERROR: {type(error).__name__}: {error} | Context: {context}")

    @classmethod
    def log_security_event(cls, logger_name: str, event: str, details: dict) -> None:
        """Log específico para eventos de segurança."""
        logger = cls.get_logger(logger_name)
        logger.warning(f"SECURITY: {event} | {details}")


# Convenience functions para uso direto
def get_logger(name: str = "eda_agent") -> logging.Logger:
    """Função de conveniência para obter logger."""
    return EDALogger.get_logger(name)


def log_data_operation(operation: str, details: dict, logger_name: str = "eda_data") -> None:
    """Log de operação de dados."""
    EDALogger.log_data_operation(logger_name, operation, details)


def log_analysis_step(analysis_type: str, status: str, metadata: Optional[dict] = None, logger_name: str = "eda_analysis") -> None:
    """Log de etapa de análise."""
    EDALogger.log_analysis_step(logger_name, analysis_type, status, metadata)


def log_llm_call(model: str, tokens_used: Optional[int] = None, duration: Optional[float] = None, logger_name: str = "eda_llm") -> None:
    """Log de chamada LLM."""
    EDALogger.log_llm_call(logger_name, model, tokens_used, duration)


def log_error_with_context(error: Exception, context: dict, logger_name: str = "eda_error") -> None:
    """Log de erro com contexto."""
    EDALogger.log_error_with_context(logger_name, error, context)


def log_security_event(event: str, details: dict, logger_name: str = "eda_security") -> None:
    """Log de evento de segurança."""
    EDALogger.log_security_event(logger_name, event, details)