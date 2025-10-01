"""
Ferramentas para análise de correlação e relacionamentos entre variáveis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from scipy import stats

from .base import BaseTool, ToolResult, ToolParameter


class CorrelationTool(BaseTool):
    """
    Ferramenta para análise de correlação entre variáveis.
    """

    def __init__(self):
        super().__init__()
        self.name = "calculate_correlation_matrix"
        self.category = "correlation_analysis"
        self.description = (
            "Calcula matriz de correlação entre variáveis numéricas usando diferentes métodos "
            "(Pearson, Spearman, Kendall). Identifica correlações fortes, moderadas e fracas."
        )
        self.parameters = [
            ToolParameter(
                name="columns",
                type="array",
                description="Lista de colunas para calcular correlação. Se não fornecida, usa todas as numéricas.",
                required=False,
                default=None
            ),
            ToolParameter(
                name="method",
                type="string",
                description="Método de correlação",
                required=False,
                default="pearson",
                enum=["pearson", "spearman", "kendall"]
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Calcula matriz de correlação.

        Args:
            df: DataFrame com os dados
            columns: Lista de colunas (opcional)
            method: Método de correlação

        Returns:
            ToolResult com matriz de correlação
        """
        start_time = datetime.now()
        columns = kwargs.get('columns')
        method = kwargs.get('method', 'pearson')

        # Selecionar colunas numéricas
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return self._create_error_result("Dataset não contém colunas numéricas.")
            columns = numeric_df.columns.tolist()
        else:
            # Verificar se colunas existem e são numéricas
            missing = [col for col in columns if col not in df.columns]
            if missing:
                return self._create_error_result(f"Colunas não encontradas: {missing}")

            non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return self._create_error_result(
                    f"Colunas não numéricas: {non_numeric}. Correlação requer colunas numéricas."
                )

        # Verificar se há pelo menos 2 colunas
        if len(columns) < 2:
            return self._create_error_result(
                "Correlação requer pelo menos 2 colunas numéricas."
            )

        # Calcular matriz de correlação
        try:
            corr_matrix = df[columns].corr(method=method)

            # Encontrar correlações mais fortes (excluindo diagonal)
            correlations_list = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]

                    if pd.notna(corr_value):
                        correlations_list.append({
                            'variable_1': col1,
                            'variable_2': col2,
                            'correlation': round(float(corr_value), 4),
                            'abs_correlation': round(abs(float(corr_value)), 4),
                            'strength': self._classify_correlation_strength(corr_value),
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        })

            # Ordenar por correlação absoluta (mais forte primeiro)
            correlations_list.sort(key=lambda x: x['abs_correlation'], reverse=True)

            # Estatísticas da matriz
            corr_values = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if pd.notna(val):
                        corr_values.append(abs(val))

            result_data = {
                'method': method,
                'correlation_matrix': corr_matrix.to_dict(),
                'correlations': correlations_list[:20],  # Top 20
                'strongest_correlation': correlations_list[0] if correlations_list else None,
                'statistics': {
                    'mean_correlation': round(float(np.mean(corr_values)), 4) if corr_values else 0,
                    'max_correlation': round(float(np.max(corr_values)), 4) if corr_values else 0,
                    'min_correlation': round(float(np.min(corr_values)), 4) if corr_values else 0,
                    'strong_correlations_count': sum(1 for v in corr_values if v > 0.7),
                    'moderate_correlations_count': sum(1 for v in corr_values if 0.3 < v <= 0.7),
                    'weak_correlations_count': sum(1 for v in corr_values if v <= 0.3)
                },
                'columns_analyzed': columns
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'method': method, 'columns_count': len(columns)}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao calcular correlação: {str(e)}")

    def _classify_correlation_strength(self, corr_value: float) -> str:
        """Classifica força da correlação."""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        else:
            return "weak"


class PairwiseCorrelationTool(BaseTool):
    """
    Ferramenta para análise de correlação entre dois pares de variáveis específicas.
    """

    def __init__(self):
        super().__init__()
        self.name = "analyze_pairwise_correlation"
        self.category = "correlation_analysis"
        self.description = (
            "Analisa correlação detalhada entre duas variáveis específicas, "
            "incluindo teste de significância estatística e força da relação."
        )
        self.parameters = [
            ToolParameter(
                name="column1",
                type="string",
                description="Nome da primeira variável",
                required=True
            ),
            ToolParameter(
                name="column2",
                type="string",
                description="Nome da segunda variável",
                required=True
            ),
            ToolParameter(
                name="method",
                type="string",
                description="Método de correlação",
                required=False,
                default="pearson",
                enum=["pearson", "spearman", "kendall"]
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Analisa correlação entre duas variáveis.

        Args:
            df: DataFrame com os dados
            column1: Primeira coluna
            column2: Segunda coluna
            method: Método de correlação

        Returns:
            ToolResult com análise de correlação
        """
        start_time = datetime.now()
        column1 = kwargs.get('column1')
        column2 = kwargs.get('column2')
        method = kwargs.get('method', 'pearson')

        # Verificar se colunas existem
        if column1 not in df.columns:
            return self._create_error_result(f"Coluna '{column1}' não encontrada.")
        if column2 not in df.columns:
            return self._create_error_result(f"Coluna '{column2}' não encontrada.")

        # Verificar se são numéricas
        if not pd.api.types.is_numeric_dtype(df[column1]):
            return self._create_error_result(f"Coluna '{column1}' não é numérica.")
        if not pd.api.types.is_numeric_dtype(df[column2]):
            return self._create_error_result(f"Coluna '{column2}' não é numérica.")

        # Remover valores nulos
        data_clean = df[[column1, column2]].dropna()

        if len(data_clean) < 3:
            return self._create_error_result(
                "Dados insuficientes para análise de correlação (mínimo 3 pares de valores)."
            )

        try:
            # Calcular correlação e p-value
            if method == 'pearson':
                corr_coef, p_value = stats.pearsonr(data_clean[column1], data_clean[column2])
            elif method == 'spearman':
                corr_coef, p_value = stats.spearmanr(data_clean[column1], data_clean[column2])
            elif method == 'kendall':
                corr_coef, p_value = stats.kendalltau(data_clean[column1], data_clean[column2])
            else:
                return self._create_error_result(f"Método '{method}' não suportado.")

            # Classificar correlação
            strength = self._classify_correlation_strength(corr_coef)
            direction = 'positive' if corr_coef > 0 else 'negative' if corr_coef < 0 else 'none'

            result_data = {
                'column1': column1,
                'column2': column2,
                'method': method,
                'correlation_coefficient': round(float(corr_coef), 4),
                'p_value': round(float(p_value), 6),
                'is_significant': p_value < 0.05,
                'strength': strength,
                'direction': direction,
                'sample_size': len(data_clean),
                'missing_pairs': len(df) - len(data_clean)
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'method': method}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao calcular correlação: {str(e)}")

    def _classify_correlation_strength(self, corr_value: float) -> str:
        """Classifica força da correlação."""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        else:
            return "weak"
