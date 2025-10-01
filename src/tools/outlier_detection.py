"""
Ferramentas para detecção de outliers e valores atípicos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from scipy import stats

from .base import BaseTool, ToolResult, ToolParameter


class OutlierDetectionTool(BaseTool):
    """
    Ferramenta para detecção de outliers usando método IQR.
    """

    def __init__(self):
        super().__init__()
        self.name = "detect_outliers_iqr"
        self.category = "outlier_detection"
        self.description = (
            "Detecta outliers em uma coluna numérica usando o método IQR (Interquartile Range). "
            "Identifica valores que estão além de 1.5 × IQR dos quartis Q1 e Q3."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna a analisar",
                required=True
            ),
            ToolParameter(
                name="multiplier",
                type="number",
                description="Multiplicador do IQR para definir threshold de outliers (padrão: 1.5)",
                required=False,
                default=1.5
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Detecta outliers usando método IQR.

        Args:
            df: DataFrame com os dados
            column: Nome da coluna
            multiplier: Multiplicador do IQR

        Returns:
            ToolResult com outliers detectados
        """
        start_time = datetime.now()
        column = kwargs.get('column')
        multiplier = kwargs.get('multiplier', 1.5)

        # Verificar se coluna existe
        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        # Verificar se é numérica
        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(
                f"Coluna '{column}' não é numérica. Tipo: {col_data.dtype}"
            )

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            # Calcular quartis e IQR
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            # Definir limites
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Identificar outliers
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outliers = col_data[outliers_mask]

            # Estatísticas dos outliers
            outlier_indices = outliers.index.tolist()
            outlier_values = outliers.values.tolist()

            result_data = {
                'column_name': column,
                'total_values': len(col_data),
                'outliers_count': len(outliers),
                'outliers_percentage': round(float(len(outliers) / len(col_data) * 100), 2),
                'bounds': {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr)
                },
                'outliers_summary': {
                    'min_outlier': float(outliers.min()) if len(outliers) > 0 else None,
                    'max_outlier': float(outliers.max()) if len(outliers) > 0 else None,
                    'mean_outlier': float(outliers.mean()) if len(outliers) > 0 else None
                },
                'outlier_indices': outlier_indices[:100],  # Limitar a 100
                'outlier_values': [float(v) for v in outlier_values[:100]],  # Limitar a 100
                'method': 'IQR',
                'multiplier': multiplier
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'column': column, 'method': 'IQR'}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao detectar outliers: {str(e)}")


class ZScoreOutlierTool(BaseTool):
    """
    Ferramenta para detecção de outliers usando Z-score.
    """

    def __init__(self):
        super().__init__()
        self.name = "detect_outliers_zscore"
        self.category = "outlier_detection"
        self.description = (
            "Detecta outliers usando método Z-score. "
            "Identifica valores que estão além de um número especificado de desvios padrão da média."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna a analisar",
                required=True
            ),
            ToolParameter(
                name="threshold",
                type="number",
                description="Threshold de Z-score para considerar outlier (padrão: 3.0)",
                required=False,
                default=3.0
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Detecta outliers usando Z-score.

        Args:
            df: DataFrame com os dados
            column: Nome da coluna
            threshold: Threshold de Z-score

        Returns:
            ToolResult com outliers detectados
        """
        start_time = datetime.now()
        column = kwargs.get('column')
        threshold = kwargs.get('threshold', 3.0)

        # Verificar se coluna existe
        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        # Verificar se é numérica
        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            # Calcular Z-scores
            mean = col_data.mean()
            std = col_data.std()

            if std == 0:
                return self._create_error_result(
                    f"Coluna '{column}' tem desvio padrão zero. Todos os valores são iguais."
                )

            z_scores = np.abs((col_data - mean) / std)

            # Identificar outliers
            outliers_mask = z_scores > threshold
            outliers = col_data[outliers_mask]
            outlier_z_scores = z_scores[outliers_mask]

            result_data = {
                'column_name': column,
                'total_values': len(col_data),
                'outliers_count': len(outliers),
                'outliers_percentage': round(float(len(outliers) / len(col_data) * 100), 2),
                'statistics': {
                    'mean': float(mean),
                    'std': float(std),
                    'threshold': threshold
                },
                'outliers_summary': {
                    'min_outlier': float(outliers.min()) if len(outliers) > 0 else None,
                    'max_outlier': float(outliers.max()) if len(outliers) > 0 else None,
                    'max_zscore': float(outlier_z_scores.max()) if len(outliers) > 0 else None
                },
                'outlier_indices': outliers.index.tolist()[:100],
                'outlier_values': [float(v) for v in outliers.values[:100]],
                'outlier_zscores': [float(z) for z in outlier_z_scores.values[:100]],
                'method': 'Z-score',
                'threshold': threshold
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'column': column, 'method': 'Z-score'}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao detectar outliers: {str(e)}")


class OutlierImpactTool(BaseTool):
    """
    Ferramenta para analisar impacto de outliers nas estatísticas.
    """

    def __init__(self):
        super().__init__()
        self.name = "analyze_outlier_impact"
        self.category = "outlier_detection"
        self.description = (
            "Analisa o impacto dos outliers nas estatísticas descritivas, "
            "comparando métricas com e sem outliers."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna a analisar",
                required=True
            ),
            ToolParameter(
                name="method",
                type="string",
                description="Método de detecção de outliers",
                required=False,
                default="iqr",
                enum=["iqr", "zscore"]
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """
        Analisa impacto de outliers.

        Args:
            df: DataFrame com os dados
            column: Nome da coluna
            method: Método de detecção

        Returns:
            ToolResult com análise de impacto
        """
        start_time = datetime.now()
        column = kwargs.get('column')
        method = kwargs.get('method', 'iqr')

        # Verificar se coluna existe
        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            # Identificar outliers baseado no método
            if method == 'iqr':
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            else:  # zscore
                mean = col_data.mean()
                std = col_data.std()
                if std == 0:
                    return self._create_error_result("Desvio padrão zero.")
                z_scores = np.abs((col_data - mean) / std)
                outliers_mask = z_scores > 3.0

            # Dados sem outliers
            data_without_outliers = col_data[~outliers_mask]

            # Calcular estatísticas com e sem outliers
            with_outliers = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max())
            }

            without_outliers = {
                'mean': float(data_without_outliers.mean()),
                'median': float(data_without_outliers.median()),
                'std': float(data_without_outliers.std()),
                'min': float(data_without_outliers.min()),
                'max': float(data_without_outliers.max())
            }

            # Calcular diferenças
            impact = {
                'mean_change': round(float(with_outliers['mean'] - without_outliers['mean']), 4),
                'median_change': round(float(with_outliers['median'] - without_outliers['median']), 4),
                'std_change': round(float(with_outliers['std'] - without_outliers['std']), 4),
                'range_change': round(float((with_outliers['max'] - with_outliers['min']) -
                                           (without_outliers['max'] - without_outliers['min'])), 4)
            }

            result_data = {
                'column_name': column,
                'outliers_count': int(outliers_mask.sum()),
                'with_outliers': with_outliers,
                'without_outliers': without_outliers,
                'impact': impact,
                'method': method
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time,
                metadata={'column': column, 'method': method}
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao analisar impacto: {str(e)}")
