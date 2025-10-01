"""
Ferramenta de análise de schema de dataset.

Esta ferramenta analisa a estrutura do dataset e classifica variáveis
em categorias conceituais (numéricas, categóricas, booleanas, temporais).
Não realiza cálculos estatísticos, apenas interpretação de tipos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from .base import BaseTool, ToolResult, ToolParameter


class SchemaAnalysisTool(BaseTool):
    """
    Analisa o schema do dataset e classifica tipos de variáveis.

    Retorna informações estruturais sobre o dataset:
    - Classificação de variáveis (numéricas contínuas, discretas, categóricas, booleanas)
    - Tipos de dados pandas
    - Informações de shape e colunas
    """

    def __init__(self):
        super().__init__()
        self.name = "analyze_dataset_schema"
        self.category = "schema_analysis"
        self.description = (
            "Analisa a estrutura e schema do dataset, classificando variáveis em categorias "
            "conceituais (numéricas contínuas, numéricas discretas, categóricas, booleanas, temporais). "
            "Use esta ferramenta para queries sobre tipos de dados, estrutura do dataset, "
            "ou classificação de variáveis. NÃO calcula estatísticas, apenas interpreta tipos."
        )
        self.parameters = []  # Não requer parâmetros, analisa dataset inteiro

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()

        try:
            # Informações básicas
            total_rows = len(df)
            total_columns = len(df.columns)

            # Classificar cada coluna
            numeric_continuous = []
            numeric_discrete = []
            categorical = []
            boolean_vars = []
            temporal = []
            other = []

            for col in df.columns:
                col_type = df[col].dtype
                col_data = df[col].dropna()

                # Verificar se é temporal
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    temporal.append(col)
                # Verificar se é booleana
                elif pd.api.types.is_bool_dtype(col_data) or set(col_data.unique()).issubset({0, 1, True, False}):
                    boolean_vars.append(col)
                # Verificar se é numérica
                elif pd.api.types.is_numeric_dtype(col_data):
                    # Distinguir entre contínua e discreta
                    unique_count = col_data.nunique()
                    total_count = len(col_data)

                    # Heurística: se tem poucos valores únicos OU todos são inteiros
                    if unique_count <= 10 or (col_type in ['int64', 'int32', 'int16', 'int8'] and unique_count < total_count * 0.05):
                        numeric_discrete.append(col)
                    else:
                        numeric_continuous.append(col)
                # Verificar se é categórica (object, string, ou numérica com poucos valores)
                elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
                    categorical.append(col)
                else:
                    other.append(col)

            # Preparar resultado estruturado
            result_data = {
                'dataset_info': {
                    'total_rows': total_rows,
                    'total_columns': total_columns,
                    'columns': list(df.columns)
                },
                'variable_classification': {
                    'numeric_continuous': {
                        'columns': numeric_continuous,
                        'count': len(numeric_continuous),
                        'dtypes': {col: str(df[col].dtype) for col in numeric_continuous}
                    },
                    'numeric_discrete': {
                        'columns': numeric_discrete,
                        'count': len(numeric_discrete),
                        'dtypes': {col: str(df[col].dtype) for col in numeric_discrete}
                    },
                    'categorical': {
                        'columns': categorical,
                        'count': len(categorical),
                        'dtypes': {col: str(df[col].dtype) for col in categorical}
                    },
                    'boolean': {
                        'columns': boolean_vars,
                        'count': len(boolean_vars),
                        'dtypes': {col: str(df[col].dtype) for col in boolean_vars}
                    },
                    'temporal': {
                        'columns': temporal,
                        'count': len(temporal),
                        'dtypes': {col: str(df[col].dtype) for col in temporal}
                    },
                    'other': {
                        'columns': other,
                        'count': len(other),
                        'dtypes': {col: str(df[col].dtype) for col in other}
                    }
                },
                'all_dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'summary': {
                    'total_numeric': len(numeric_continuous) + len(numeric_discrete),
                    'total_categorical': len(categorical),
                    'total_boolean': len(boolean_vars),
                    'total_temporal': len(temporal),
                    'total_other': len(other)
                }
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao analisar schema: {str(e)}")
