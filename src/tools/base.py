"""
Classes base para ferramentas estatísticas.
Define interface comum para todas as ferramentas do sistema.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd


@dataclass
class ToolParameter:
    """Definição de um parâmetro de ferramenta."""
    name: str
    type: str  # 'string', 'number', 'boolean', 'array', 'object'
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None  # Valores permitidos


@dataclass
class ToolResult:
    """Resultado da execução de uma ferramenta."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converte resultado para dicionário."""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class BaseTool(ABC):
    """
    Classe base abstrata para todas as ferramentas estatísticas.

    Todas as ferramentas devem:
    1. Herdar desta classe
    2. Implementar método execute()
    3. Definir name, description e parameters
    """

    def __init__(self):
        self.name: str = self.__class__.__name__
        self.category: str = 'general'
        self.description: str = ''
        self.parameters: List[ToolParameter] = []

    @abstractmethod
    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Executa a ferramenta com os parâmetros fornecidos.

        Args:
            df: DataFrame com os dados
            **kwargs: Parâmetros específicos da ferramenta

        Returns:
            ToolResult com resultado da execução
        """
        pass

    def get_description(self) -> Dict[str, Any]:
        """
        Retorna descrição estruturada da ferramenta para o LLM.

        Returns:
            Dicionário com metadados da ferramenta
        """
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'parameters': [
                {
                    'name': p.name,
                    'type': p.type,
                    'description': p.description,
                    'required': p.required,
                    'default': p.default,
                    'enum': p.enum
                }
                for p in self.parameters
            ]
        }

    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Valida se os parâmetros fornecidos estão corretos.

        Args:
            **kwargs: Parâmetros a validar

        Returns:
            Tupla (válido, mensagem_erro)
        """
        for param in self.parameters:
            # Verificar parâmetros obrigatórios
            if param.required and param.name not in kwargs:
                return False, f"Parâmetro obrigatório ausente: {param.name}"

            # Verificar valores enum
            if param.enum and param.name in kwargs:
                if kwargs[param.name] not in param.enum:
                    return False, f"Valor inválido para {param.name}. Valores permitidos: {param.enum}"

        return True, None

    def _create_error_result(self, error_message: str) -> ToolResult:
        """
        Cria resultado de erro padronizado.

        Args:
            error_message: Mensagem de erro

        Returns:
            ToolResult com erro
        """
        return ToolResult(
            success=False,
            data={},
            error=error_message,
            metadata={'timestamp': datetime.now().isoformat()}
        )

    def _create_success_result(
        self,
        data: Dict[str, Any],
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Cria resultado de sucesso padronizado.

        Args:
            data: Dados do resultado
            execution_time: Tempo de execução em segundos
            metadata: Metadados adicionais

        Returns:
            ToolResult com sucesso
        """
        result_metadata = metadata or {}
        result_metadata['timestamp'] = datetime.now().isoformat()

        return ToolResult(
            success=True,
            data=data,
            error=None,
            execution_time=execution_time,
            metadata=result_metadata
        )
