"""
Ferramentas para análise de dados ausentes (missing data).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseTool, ToolResult, ToolParameter


class MissingDataTool(BaseTool):
    """
    Ferramenta para análise de padrões de dados ausentes.
    """

    def __init__(self):
        super().__init__()
        self.name = "analyze_missing_data"
        self.category = "data_quality"
        self.description = (
            "Analisa padrões de dados ausentes no dataset, identificando colunas com missing values, "
            "percentuais de ausência e padrões de distribuição dos dados faltantes."
        )
        self.parameters = []  # Não requer parâmetros, analisa dataset inteiro

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Analisa padrões de dados ausentes.

        Args:
            df: DataFrame com os dados

        Returns:
            ToolResult com análise de missing data
        """
        start_time = datetime.now()

        try:
            # Contar valores ausentes por coluna
            missing_counts = df.isnull().sum()
            total_rows = len(df)

            # Colunas com dados ausentes
            columns_with_missing = []
            for column in df.columns:
                missing_count = missing_counts[column]
                if missing_count > 0:
                    columns_with_missing.append({
                        'column': column,
                        'missing_count': int(missing_count),
                        'missing_percentage': round(float(missing_count / total_rows * 100), 2),
                        'data_type': str(df[column].dtype)
                    })

            # Ordenar por percentual de ausência
            columns_with_missing.sort(key=lambda x: x['missing_percentage'], reverse=True)

            # Análise de linhas com dados ausentes
            rows_with_any_missing = df.isnull().any(axis=1).sum()
            rows_completely_missing = df.isnull().all(axis=1).sum()
            complete_rows = total_rows - rows_with_any_missing

            # Estatísticas globais
            total_missing = missing_counts.sum()
            total_cells = df.size

            result_data = {
                'total_columns': len(df.columns),
                'columns_with_missing': len(columns_with_missing),
                'columns_without_missing': len(df.columns) - len(columns_with_missing),
                'total_rows': total_rows,
                'rows_with_missing': int(rows_with_any_missing),
                'complete_rows': int(complete_rows),
                'rows_completely_missing': int(rows_completely_missing),
                'overall_statistics': {
                    'total_missing_values': int(total_missing),
                    'total_cells': int(total_cells),
                    'missing_percentage': round(float(total_missing / total_cells * 100), 2),
                    'completeness_percentage': round(float(100 - (total_missing / total_cells * 100)), 2)
                },
                'columns_details': columns_with_missing,
                'most_affected_columns': columns_with_missing[:10]  # Top 10
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'dataset_shape': df.shape}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao analisar dados ausentes: {str(e)}")


class MissingDataColumnTool(BaseTool):
    """
    Ferramenta para análise detalhada de dados ausentes em uma coluna específica.
    """

    def __init__(self):
        super().__init__()
        self.name = "analyze_column_missing_data"
        self.category = "data_quality"
        self.description = (
            "Analisa detalhadamente os dados ausentes em uma coluna específica, "
            "incluindo distribuição e padrões de ausência."
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
        Analisa dados ausentes em uma coluna específica.

        Args:
            df: DataFrame com os dados
            column: Nome da coluna

        Returns:
            ToolResult com análise detalhada
        """
        start_time = datetime.now()
        column = kwargs.get('column')

        # Verificar se coluna existe
        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        try:
            col_data = df[column]
            missing_count = col_data.isnull().sum()
            total_count = len(col_data)

            # Índices com valores ausentes
            missing_indices = col_data[col_data.isnull()].index.tolist()

            # Estatísticas
            result_data = {
                'column_name': column,
                'data_type': str(col_data.dtype),
                'total_values': total_count,
                'missing_count': int(missing_count),
                'present_count': int(total_count - missing_count),
                'missing_percentage': round(float(missing_count / total_count * 100), 2),
                'missing_indices': missing_indices[:100],  # Primeiros 100
                'has_missing': missing_count > 0
            }

            # Se a coluna tem valores, adicionar estatísticas dos valores presentes
            if missing_count < total_count:
                present_data = col_data.dropna()

                if pd.api.types.is_numeric_dtype(present_data):
                    result_data['present_values_stats'] = {
                        'mean': float(present_data.mean()),
                        'median': float(present_data.median()),
                        'std': float(present_data.std()) if len(present_data) > 1 else 0.0,
                        'min': float(present_data.min()),
                        'max': float(present_data.max())
                    }
                else:
                    result_data['present_values_stats'] = {
                        'unique_values': int(present_data.nunique()),
                        'most_common': str(present_data.mode()[0]) if not present_data.mode().empty else None
                    }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'column': column}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao analisar coluna: {str(e)}")


class MissingDataCorrelationTool(BaseTool):
    """
    Ferramenta para analisar correlação entre padrões de missing data.
    """

    def __init__(self):
        super().__init__()
        self.name = "analyze_missing_data_correlation"
        self.category = "data_quality"
        self.description = (
            "Analisa se há correlação entre padrões de dados ausentes em diferentes colunas, "
            "ajudando a identificar se dados estão faltando de forma sistemática."
        )
        self.parameters = []

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Analisa correlação entre padrões de missing data.

        Args:
            df: DataFrame com os dados

        Returns:
            ToolResult com análise de correlação
        """
        start_time = datetime.now()

        try:
            # Criar DataFrame de indicadores de missing (True/False)
            missing_indicators = df.isnull()

            # Selecionar apenas colunas que têm dados ausentes
            columns_with_missing = missing_indicators.columns[missing_indicators.any()].tolist()

            if len(columns_with_missing) < 2:
                return self._create_error_result(
                    "Dataset não tem colunas suficientes com missing data para análise de correlação."
                )

            # Calcular correlação entre padrões de missing
            missing_corr = missing_indicators[columns_with_missing].astype(int).corr()

            # Encontrar pares com alta correlação
            correlated_pairs = []
            for i in range(len(missing_corr.columns)):
                for j in range(i + 1, len(missing_corr.columns)):
                    col1 = missing_corr.columns[i]
                    col2 = missing_corr.columns[j]
                    corr_value = missing_corr.iloc[i, j]

                    if pd.notna(corr_value) and abs(corr_value) > 0.3:
                        correlated_pairs.append({
                            'column_1': col1,
                            'column_2': col2,
                            'correlation': round(float(corr_value), 4)
                        })

            correlated_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

            result_data = {
                'columns_analyzed': columns_with_missing,
                'correlated_pairs_count': len(correlated_pairs),
                'correlated_pairs': correlated_pairs[:20],  # Top 20
                'correlation_matrix': missing_corr.to_dict() if len(columns_with_missing) <= 20 else None
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'columns_count': len(columns_with_missing)}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao analisar correlação: {str(e)}")
