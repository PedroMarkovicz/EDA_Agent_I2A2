"""
Ferramentas estatísticas minimalistas para valores únicos.

Estas ferramentas retornam APENAS o valor solicitado, sem informações extras.
Ideal para queries específicas como "Qual a média?" ou "Qual o máximo?".
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

from .base import BaseTool, ToolResult, ToolParameter


class GetMeanTool(BaseTool):
    """Retorna APENAS a média de uma coluna."""

    def __init__(self):
        super().__init__()
        self.name = "get_mean"
        self.category = "minimal_statistics"
        self.description = (
            "Calcula e retorna APENAS a média (mean) de uma coluna numérica. "
            "Use esta ferramenta quando o usuário pedir especificamente a média e nada mais."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para calcular a média",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            mean_value = float(col_data.mean())
            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data={
                    'column': column,
                    'mean': mean_value
                },
                execution_time=execution_time
            )
        except Exception as e:
            return self._create_error_result(f"Erro ao calcular média: {str(e)}")


class GetMedianTool(BaseTool):
    """Retorna APENAS a mediana de uma coluna."""

    def __init__(self):
        super().__init__()
        self.name = "get_median"
        self.category = "minimal_statistics"
        self.description = (
            "Calcula e retorna APENAS a mediana de uma coluna numérica. "
            "Use esta ferramenta quando o usuário pedir especificamente a mediana e nada mais."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para calcular a mediana",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            median_value = float(col_data.median())
            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data={
                    'column': column,
                    'median': median_value
                },
                execution_time=execution_time
            )
        except Exception as e:
            return self._create_error_result(f"Erro ao calcular mediana: {str(e)}")


class GetMaxTool(BaseTool):
    """Retorna APENAS o valor máximo de uma coluna."""

    def __init__(self):
        super().__init__()
        self.name = "get_max"
        self.category = "minimal_statistics"
        self.description = (
            "Retorna APENAS o valor máximo de uma coluna numérica. "
            "Use esta ferramenta quando o usuário pedir especificamente o máximo e nada mais."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para encontrar o máximo",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            max_value = float(col_data.max())
            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data={
                    'column': column,
                    'max': max_value
                },
                execution_time=execution_time
            )
        except Exception as e:
            return self._create_error_result(f"Erro ao calcular máximo: {str(e)}")


class GetMinTool(BaseTool):
    """Retorna APENAS o valor mínimo de uma coluna."""

    def __init__(self):
        super().__init__()
        self.name = "get_min"
        self.category = "minimal_statistics"
        self.description = (
            "Retorna APENAS o valor mínimo de uma coluna numérica. "
            "Use esta ferramenta quando o usuário pedir especificamente o mínimo e nada mais."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para encontrar o mínimo",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            min_value = float(col_data.min())
            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data={
                    'column': column,
                    'min': min_value
                },
                execution_time=execution_time
            )
        except Exception as e:
            return self._create_error_result(f"Erro ao calcular mínimo: {str(e)}")


class GetSumTool(BaseTool):
    """Retorna APENAS a soma dos valores de uma coluna."""

    def __init__(self):
        super().__init__()
        self.name = "get_sum"
        self.category = "minimal_statistics"
        self.description = (
            "Calcula e retorna APENAS a soma total de uma coluna numérica. "
            "Use esta ferramenta quando o usuário pedir especificamente a soma e nada mais."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para calcular a soma",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            sum_value = float(col_data.sum())
            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data={
                    'column': column,
                    'sum': sum_value
                },
                execution_time=execution_time
            )
        except Exception as e:
            return self._create_error_result(f"Erro ao calcular soma: {str(e)}")


class GetStdTool(BaseTool):
    """Retorna APENAS o desvio padrão de uma coluna."""

    def __init__(self):
        super().__init__()
        self.name = "get_std"
        self.category = "minimal_statistics"
        self.description = (
            "Calcula e retorna APENAS o desvio padrão de uma coluna numérica. "
            "Use esta ferramenta quando o usuário pedir especificamente o desvio padrão e nada mais."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para calcular o desvio padrão",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        if len(col_data) < 2:
            return self._create_error_result(f"Desvio padrão requer pelo menos 2 valores.")

        try:
            std_value = float(col_data.std())
            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data={
                    'column': column,
                    'std': std_value
                },
                execution_time=execution_time
            )
        except Exception as e:
            return self._create_error_result(f"Erro ao calcular desvio padrão: {str(e)}")


class GetCountTool(BaseTool):
    """Retorna APENAS a contagem de valores válidos de uma coluna."""

    def __init__(self):
        super().__init__()
        self.name = "get_count"
        self.category = "minimal_statistics"
        self.description = (
            "Retorna APENAS a contagem de valores válidos (não-nulos) de uma coluna. "
            "Use esta ferramenta quando o usuário pedir especificamente a contagem e nada mais."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para contar valores",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        try:
            count_value = int(df[column].count())
            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data={
                    'column': column,
                    'count': count_value
                },
                execution_time=execution_time
            )
        except Exception as e:
            return self._create_error_result(f"Erro ao contar valores: {str(e)}")
