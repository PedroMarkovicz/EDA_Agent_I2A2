"""
Ferramentas para estatísticas descritivas básicas.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseTool, ToolResult, ToolParameter


class BasicStatsTool(BaseTool):
    """
    Ferramenta para cálculo de estatísticas descritivas básicas.

    Fornece métricas como média, mediana, desvio padrão, quartis,
    valores mínimos e máximos para colunas numéricas.
    """

    def __init__(self):
        super().__init__()
        self.name = "calculate_descriptive_stats"
        self.category = "basic_statistics"
        self.description = (
            "Calcula estatísticas descritivas completas para uma coluna numérica, "
            "incluindo média, mediana, desvio padrão, valores mínimo e máximo, "
            "quartis e contagem de valores válidos."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna a analisar",
                required=True
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Executa cálculo de estatísticas descritivas.

        Args:
            df: DataFrame com os dados
            column: Nome da coluna a analisar

        Returns:
            ToolResult com estatísticas calculadas
        """
        start_time = datetime.now()
        column = kwargs.get('column')

        # Verificar se coluna existe
        if column not in df.columns:
            return self._create_error_result(
                f"Coluna '{column}' não encontrada no dataset. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        col_data = df[column]

        # Remover valores nulos para cálculos
        col_data_clean = col_data.dropna()

        if col_data_clean.empty:
            return self._create_error_result(
                f"Coluna '{column}' não contém valores válidos (todos são nulos)."
            )

        # Verificar se é numérica
        if not pd.api.types.is_numeric_dtype(col_data_clean):
            return self._create_error_result(
                f"Coluna '{column}' não é numérica. "
                f"Tipo detectado: {col_data.dtype}. "
                f"Use ferramentas de análise categórica para este tipo de dado."
            )

        # Calcular estatísticas
        try:
            stats = {
                'column_name': column,
                'count': int(col_data_clean.count()),
                'missing_count': int(col_data.isnull().sum()),
                'missing_percentage': round(float(col_data.isnull().sum() / len(df) * 100), 2),
                'mean': float(col_data_clean.mean()),
                'median': float(col_data_clean.median()),
                'mode': float(col_data_clean.mode()[0]) if not col_data_clean.mode().empty else None,
                'std': float(col_data_clean.std()) if len(col_data_clean) > 1 else 0.0,
                'variance': float(col_data_clean.var()) if len(col_data_clean) > 1 else 0.0,
                'min': float(col_data_clean.min()),
                'max': float(col_data_clean.max()),
                'range': float(col_data_clean.max() - col_data_clean.min()),
                'q25': float(col_data_clean.quantile(0.25)),
                'q50': float(col_data_clean.quantile(0.50)),  # Mesmo que mediana
                'q75': float(col_data_clean.quantile(0.75)),
                'iqr': float(col_data_clean.quantile(0.75) - col_data_clean.quantile(0.25)),
                'sum': float(col_data_clean.sum()),
                'skewness': float(col_data_clean.skew()) if len(col_data_clean) > 2 else None,
                'kurtosis': float(col_data_clean.kurtosis()) if len(col_data_clean) > 3 else None,
            }

            # Adicionar informações sobre a distribuição
            if stats['std'] > 0:
                stats['coefficient_of_variation'] = round(float(stats['std'] / stats['mean'] * 100), 2) if stats['mean'] != 0 else None
            else:
                stats['coefficient_of_variation'] = 0.0

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=stats,
                execution_time=execution_time,
                metadata={
                    'column': column,
                    'data_type': str(col_data.dtype),
                    'calculation_method': 'pandas'
                }
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao calcular estatísticas: {str(e)}")


class SummaryStatisticsTool(BaseTool):
    """
    Ferramenta para resumo estatístico de múltiplas colunas.
    """

    def __init__(self):
        super().__init__()
        self.name = "calculate_summary_statistics"
        self.category = "basic_statistics"
        self.description = (
            "Calcula resumo estatístico para múltiplas colunas numéricas simultaneamente, "
            "retornando estatísticas principais (média, mediana, desvio padrão) para cada coluna."
        )
        self.parameters = [
            ToolParameter(
                name="columns",
                type="array",
                description="Lista de nomes das colunas a analisar. Se não fornecida, analisa todas as colunas numéricas.",
                required=False,
                default=None
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Executa cálculo de resumo estatístico para múltiplas colunas.

        Args:
            df: DataFrame com os dados
            columns: Lista de colunas (opcional)

        Returns:
            ToolResult com resumo estatístico
        """
        start_time = datetime.now()
        columns = kwargs.get('columns')

        # Se colunas não fornecidas, usar todas as numéricas
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return self._create_error_result(
                    "Dataset não contém colunas numéricas para análise."
                )
            columns = numeric_cols
        else:
            # Verificar se todas as colunas existem
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return self._create_error_result(
                    f"Colunas não encontradas: {missing_cols}"
                )

        # Calcular estatísticas para cada coluna
        summary = {}
        for column in columns:
            col_data = df[column].dropna()

            if not pd.api.types.is_numeric_dtype(col_data):
                continue  # Pular colunas não-numéricas

            if col_data.empty:
                continue  # Pular colunas vazias

            summary[column] = {
                'count': int(col_data.count()),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                'min': float(col_data.min()),
                'max': float(col_data.max()),
            }

        if not summary:
            return self._create_error_result(
                "Nenhuma coluna numérica válida foi encontrada para análise."
            )

        execution_time = (datetime.now() - start_time).total_seconds()

        return self._create_success_result(
            data={'summary': summary, 'columns_analyzed': list(summary.keys())},
            execution_time=execution_time,
            metadata={'total_columns': len(summary)}
        )


class ValueCountsTool(BaseTool):
    """
    Ferramenta para contagem de valores únicos.
    """

    def __init__(self):
        super().__init__()
        self.name = "count_unique_values"
        self.category = "basic_statistics"
        self.description = (
            "Conta a frequência de cada valor único em uma coluna, "
            "retornando os valores mais comuns e suas contagens."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna a analisar",
                required=True
            ),
            ToolParameter(
                name="top_n",
                type="number",
                description="Número de valores mais frequentes a retornar",
                required=False,
                default=10
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Executa contagem de valores únicos.

        Args:
            df: DataFrame com os dados
            column: Nome da coluna
            top_n: Número de valores mais frequentes a retornar

        Returns:
            ToolResult com contagens
        """
        start_time = datetime.now()
        column = kwargs.get('column')
        top_n = kwargs.get('top_n', 10)

        # Verificar se coluna existe
        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        # Calcular contagens
        value_counts = col_data.value_counts()
        total_unique = len(value_counts)

        # Pegar top N
        top_values = value_counts.head(top_n)

        result_data = {
            'column_name': column,
            'total_unique_values': total_unique,
            'total_values': int(col_data.count()),
            'top_values': [
                {
                    'value': str(value),
                    'count': int(count),
                    'percentage': round(float(count / len(col_data) * 100), 2)
                }
                for value, count in top_values.items()
            ],
            'missing_count': int(df[column].isnull().sum())
        }

        execution_time = (datetime.now() - start_time).total_seconds()

        return self._create_success_result(
            data=result_data,
            execution_time=execution_time,
            metadata={'column': column, 'top_n': top_n}
        )
