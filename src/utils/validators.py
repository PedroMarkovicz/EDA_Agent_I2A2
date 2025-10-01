"""
Validadores para dados CSV, consultas de usuario e entrada segura.
Garante integridade dos dados e seguranca das operacoes antes da execucao.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import ast

from ..core.config import get_config
from ..core.logger import get_logger, log_data_operation, log_error_with_context, log_security_event
from ..models.enums import EDAAnalysisType


class ValidationError(Exception):
    """Erro especifico de validacao."""
    pass


class EDADataValidator:
    """Validador para dados e consultas EDA."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("data_validator")

    def validate_dataframe(self, df: pd.DataFrame, context: str = "general") -> Dict[str, Any]:
        """
        Valida um DataFrame para operacoes EDA.

        Args:
            df: DataFrame a ser validado
            context: Contexto da validacao

        Returns:
            Dicionario com resultado da validacao
        """
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "info": {
                    "shape": df.shape,
                    "memory_mb": 0.0,
                    "columns": list(df.columns),
                    "dtypes": {},
                    "missing_data": {},
                    "duplicates": 0
                }
            }

            # Verificar se DataFrame e vazio
            if df.empty:
                validation_result["errors"].append("DataFrame esta vazio")
                validation_result["is_valid"] = False
                return validation_result

            # Calcular uso de memoria
            memory_usage = df.memory_usage(deep=True).sum()
            validation_result["info"]["memory_mb"] = memory_usage / (1024 * 1024)

            # Verificar tamanho excessivo
            if validation_result["info"]["memory_mb"] > 1000:  # 1GB
                validation_result["warnings"].append(
                    f"Dataset muito grande ({validation_result['info']['memory_mb']:.1f}MB)"
                )

            # Analisar tipos de dados
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                validation_result["info"]["dtypes"][col] = dtype_str

                # Verificar dados ausentes
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                validation_result["info"]["missing_data"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_pct)
                }

                # Alertar sobre muitos dados ausentes
                if missing_pct > 50:
                    validation_result["warnings"].append(
                        f"Coluna '{col}' tem {missing_pct:.1f}% de dados ausentes"
                    )

            # Verificar duplicatas
            duplicate_count = df.duplicated().sum()
            validation_result["info"]["duplicates"] = int(duplicate_count)

            if duplicate_count > 0:
                duplicate_pct = (duplicate_count / len(df)) * 100
                validation_result["warnings"].append(
                    f"{duplicate_count} linhas duplicadas ({duplicate_pct:.1f}%)"
                )

            # Verificar colunas com apenas um valor
            constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if constant_cols:
                validation_result["warnings"].append(
                    f"Colunas com valores constantes: {constant_cols}"
                )

            # Verificar colunas numericas com valores infinitos
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if np.isinf(df[col]).any():
                    validation_result["warnings"].append(
                        f"Coluna '{col}' contem valores infinitos"
                    )

            log_data_operation(
                "dataframe_validated",
                {
                    "context": context,
                    "shape": df.shape,
                    "warnings_count": len(validation_result["warnings"]),
                    "errors_count": len(validation_result["errors"])
                },
                "dataframe_validator"
            )

            return validation_result

        except Exception as e:
            log_error_with_context(
                e,
                {"context": context, "dataframe_shape": df.shape if 'df' in locals() else "unknown"},
                "dataframe_validation_error"
            )
            return {
                "is_valid": False,
                "errors": [f"Erro na validacao: {str(e)}"],
                "warnings": [],
                "info": {}
            }

    def validate_column_for_analysis(self,
                                   df: pd.DataFrame,
                                   column_name: str,
                                   analysis_type: EDAAnalysisType) -> Dict[str, Any]:
        """
        Valida se uma coluna e adequada para um tipo especifico de analise.

        Args:
            df: DataFrame contendo a coluna
            column_name: Nome da coluna
            analysis_type: Tipo de analise EDA

        Returns:
            Resultado da validacao
        """
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "column_info": {}
            }

            # Verificar se a coluna existe
            if column_name not in df.columns:
                validation_result["errors"].append(f"Coluna '{column_name}' nao encontrada")
                validation_result["is_valid"] = False
                return validation_result

            column = df[column_name]

            # Informacoes basicas da coluna
            validation_result["column_info"] = {
                "dtype": str(column.dtype),
                "unique_count": column.nunique(),
                "missing_count": column.isna().sum(),
                "missing_percentage": (column.isna().sum() / len(column)) * 100
            }

            # Validacoes especificas por tipo de analise
            if analysis_type == EDAAnalysisType.DESCRIPTIVE:
                self._validate_for_descriptive(column, validation_result)
            elif analysis_type == EDAAnalysisType.RELATIONSHIP:
                self._validate_for_correlation(column, validation_result)
            elif analysis_type == EDAAnalysisType.ANOMALY:
                self._validate_for_outlier_detection(column, validation_result)
            elif analysis_type == EDAAnalysisType.PATTERN:
                self._validate_for_pattern_analysis(column, validation_result)

            return validation_result

        except Exception as e:
            log_error_with_context(
                e,
                {"column_name": column_name, "analysis_type": analysis_type},
                "column_validation_error"
            )
            return {
                "is_valid": False,
                "errors": [f"Erro na validacao da coluna: {str(e)}"],
                "warnings": [],
                "column_info": {}
            }

    def _validate_for_descriptive(self, column: pd.Series, result: Dict[str, Any]):
        """Validacao para analise descritiva."""
        # Qualquer tipo de dado e valido para analise descritiva
        if column.empty:
            result["errors"].append("Coluna vazia")
            result["is_valid"] = False

        if result["column_info"]["missing_percentage"] > 90:
            result["warnings"].append("Mais de 90% dos dados estao ausentes")

    def _validate_for_correlation(self, column: pd.Series, result: Dict[str, Any]):
        """Validacao para analise de correlacao."""
        if not pd.api.types.is_numeric_dtype(column):
            result["errors"].append("Correlacao requer dados numericos")
            result["is_valid"] = False
            return

        # Verificar variancia zero
        if column.var() == 0:
            result["errors"].append("Coluna tem variancia zero (valores constantes)")
            result["is_valid"] = False

        if result["column_info"]["missing_percentage"] > 50:
            result["warnings"].append("Muitos dados ausentes podem afetar a correlacao")

    def _validate_for_outlier_detection(self, column: pd.Series, result: Dict[str, Any]):
        """Validacao para deteccao de outliers."""
        if not pd.api.types.is_numeric_dtype(column):
            result["errors"].append("Deteccao de outliers requer dados numericos")
            result["is_valid"] = False
            return

        if column.nunique() < 5:
            result["warnings"].append("Poucos valores unicos para deteccao confiavel de outliers")

        if result["column_info"]["missing_percentage"] > 30:
            result["warnings"].append("Muitos dados ausentes podem afetar a deteccao de outliers")

    def _validate_for_pattern_analysis(self, column: pd.Series, result: Dict[str, Any]):
        """Validacao para analise de padroes."""
        if column.nunique() < 3:
            result["warnings"].append("Poucos valores unicos para analise de padroes")

        if len(column) < 10:
            result["warnings"].append("Poucos pontos de dados para analise de padroes")

    def validate_user_query(self, query: str) -> Dict[str, Any]:
        """
        Valida consulta do usuario para seguranca e formato.

        Args:
            query: Consulta em linguagem natural

        Returns:
            Resultado da validacao
        """
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "query_info": {
                    "length": len(query),
                    "word_count": len(query.split()),
                    "detected_intent": None,
                    "mentioned_columns": []
                }
            }

            # Verificar tamanho da consulta
            if len(query.strip()) == 0:
                validation_result["errors"].append("Consulta vazia")
                validation_result["is_valid"] = False
                return validation_result

            if len(query) > 1000:
                validation_result["warnings"].append("Consulta muito longa")

            # Verificar caracteres suspeitos
            suspicious_patterns = [
                r'<script.*?</script>',  # JavaScript
                r'javascript:',
                r'data:',
                r'vbscript:',
                r'onload=',
                r'onerror=',
                r'eval\s*\(',
                r'exec\s*\(',
                r'import\s+os',
                r'import\s+sys',
                r'__import__',
                r'subprocess',
                r'system\s*\('
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    log_security_event(
                        "suspicious_pattern_detected",
                        {"pattern": pattern, "query": query[:100]},
                        "query_validator"
                    )
                    validation_result["errors"].append("Consulta contem conteudo potencialmente perigoso")
                    validation_result["is_valid"] = False

            # Detectar intencao da consulta
            intent_keywords = {
                "descriptive": ["media", "mediana", "resumo", "estatistica", "descritiva"],
                "correlation": ["correlacao", "relacao", "associacao", "dependencia"],
                "visualization": ["grafico", "plot", "visualizar", "mostrar", "plotar"],
                "outlier": ["outlier", "anomalia", "atipico", "excepcao"],
                "pattern": ["padrao", "tendencia", "comportamento", "evolucao"]
            }

            detected_intents = []
            for intent, keywords in intent_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    detected_intents.append(intent)

            validation_result["query_info"]["detected_intent"] = detected_intents

            log_data_operation(
                "user_query_validated",
                {
                    "query_length": len(query),
                    "detected_intents": detected_intents,
                    "is_valid": validation_result["is_valid"]
                },
                "query_validator"
            )

            return validation_result

        except Exception as e:
            log_error_with_context(
                e,
                {"query_preview": query[:100] if query else "empty"},
                "query_validation_error"
            )
            return {
                "is_valid": False,
                "errors": [f"Erro na validacao da consulta: {str(e)}"],
                "warnings": [],
                "query_info": {}
            }

    def validate_column_names(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida nomes das colunas do DataFrame.

        Args:
            df: DataFrame a ser validado

        Returns:
            Resultado da validacao
        """
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "column_issues": {}
            }

            # Verificar colunas duplicadas
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                validation_result["errors"].append(f"Colunas duplicadas: {duplicate_cols}")
                validation_result["is_valid"] = False

            # Verificar nomes problematicos
            for col in df.columns:
                issues = []

                # Verificar espacos em branco
                if col != col.strip():
                    issues.append("espacos no inicio/fim")

                # Verificar caracteres especiais problematicos
                if re.search(r'[<>:"/\\|?*]', col):
                    issues.append("caracteres especiais problematicos")

                # Verificar se comeca com numero
                if col and col[0].isdigit():
                    issues.append("comeca com numero")

                # Verificar tamanho excessivo
                if len(col) > 100:
                    issues.append("nome muito longo")

                # Verificar se e vazio
                if not col or col.isspace():
                    issues.append("nome vazio ou apenas espacos")

                if issues:
                    validation_result["column_issues"][col] = issues
                    validation_result["warnings"].append(f"Coluna '{col}': {', '.join(issues)}")

            return validation_result

        except Exception as e:
            log_error_with_context(
                e,
                {"columns_count": len(df.columns) if 'df' in locals() else 0},
                "column_names_validation_error"
            )
            return {
                "is_valid": False,
                "errors": [f"Erro na validacao dos nomes das colunas: {str(e)}"],
                "warnings": [],
                "column_issues": {}
            }

    def validate_numeric_data_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida intervalos de dados numericos para detectar valores suspeitos.

        Args:
            df: DataFrame com dados numericos

        Returns:
            Resultado da validacao
        """
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "numeric_analysis": {}
            }

            numeric_columns = df.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                column_analysis = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "has_negative": (df[col] < 0).any(),
                    "has_zero": (df[col] == 0).any(),
                    "has_infinity": np.isinf(df[col]).any(),
                    "extreme_values": False
                }

                # Detectar valores extremos (fora de 3 desvios padrao)
                if column_analysis["std"] > 0:
                    z_scores = np.abs((df[col] - column_analysis["mean"]) / column_analysis["std"])
                    extreme_count = (z_scores > 3).sum()
                    if extreme_count > 0:
                        column_analysis["extreme_values"] = True
                        validation_result["warnings"].append(
                            f"Coluna '{col}' tem {extreme_count} valores extremos"
                        )

                # Verificar valores infinitos
                if column_analysis["has_infinity"]:
                    validation_result["warnings"].append(f"Coluna '{col}' contem valores infinitos")

                # Verificar intervalos suspeitos
                value_range = column_analysis["max"] - column_analysis["min"]
                if value_range == 0:
                    validation_result["warnings"].append(f"Coluna '{col}' tem valores constantes")

                validation_result["numeric_analysis"][col] = column_analysis

            return validation_result

        except Exception as e:
            log_error_with_context(
                e,
                {"numeric_columns": len(numeric_columns) if 'numeric_columns' in locals() else 0},
                "numeric_ranges_validation_error"
            )
            return {
                "is_valid": False,
                "errors": [f"Erro na validacao de intervalos numericos: {str(e)}"],
                "warnings": [],
                "numeric_analysis": {}
            }


# Instancia singleton
_data_validator: Optional[EDADataValidator] = None


def get_data_validator() -> EDADataValidator:
    """Obtem instancia singleton do validador de dados."""
    global _data_validator
    if _data_validator is None:
        _data_validator = EDADataValidator()
    return _data_validator


# Funcoes de conveniencia
def validate_dataframe(df: pd.DataFrame, context: str = "general") -> Dict[str, Any]:
    """Funcao de conveniencia para validacao de DataFrame."""
    return get_data_validator().validate_dataframe(df, context)


def validate_user_query(query: str) -> Dict[str, Any]:
    """Funcao de conveniencia para validacao de consulta."""
    return get_data_validator().validate_user_query(query)


def validate_column_for_analysis(df: pd.DataFrame,
                                column_name: str,
                                analysis_type: EDAAnalysisType) -> Dict[str, Any]:
    """Funcao de conveniencia para validacao de coluna para analise."""
    return get_data_validator().validate_column_for_analysis(df, column_name, analysis_type)


def is_dataframe_valid(df: pd.DataFrame) -> bool:
    """Funcao simplificada para verificar se DataFrame e valido."""
    result = validate_dataframe(df)
    return result["is_valid"]


def is_query_safe(query: str) -> bool:
    """Funcao simplificada para verificar se consulta e segura."""
    result = validate_user_query(query)
    return result["is_valid"]


class CSVValidator:
    """
    Validador específico para arquivos CSV na interface Streamlit.

    Fornece validações simples e rápidas para uploads de CSV,
    focando na usabilidade da interface web.
    """

    def __init__(self):
        self.logger = get_logger("csv_validator")
        self.data_validator = get_data_validator()

    def validate_csv_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida dados CSV carregados via Streamlit.

        Args:
            df: DataFrame com os dados CSV

        Returns:
            Resultado da validação com is_valid, issues, warnings
        """
        try:
            result = {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                }
            }

            # Validações básicas
            if df.empty:
                result["is_valid"] = False
                result["issues"].append("Arquivo CSV está vazio")
                return result

            # Validar tamanho
            if len(df) > 100000:  # 100k rows
                result["warnings"].append("Arquivo muito grande (>100k linhas). Performance pode ser afetada.")

            if len(df.columns) > 100:  # 100 columns
                result["warnings"].append("Muitas colunas (>100). Performance pode ser afetada.")

            # Verificar se há pelo menos uma coluna numérica para análises
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                result["warnings"].append("Nenhuma coluna numérica encontrada. Algumas análises podem ser limitadas.")

            # Verificar colunas com nomes problemáticos
            problematic_columns = []
            for col in df.columns:
                if not col or col.isspace():
                    problematic_columns.append("coluna com nome vazio")
                elif col.startswith(' ') or col.endswith(' '):
                    problematic_columns.append(f"'{col}' (espaços no início/fim)")

            if problematic_columns:
                result["warnings"].append(f"Colunas com nomes problemáticos: {', '.join(problematic_columns)}")

            # Verificar dados ausentes excessivos
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 50:
                result["warnings"].append(f"Muitos dados ausentes ({missing_percentage:.1f}% do total)")
            elif missing_percentage > 20:
                result["warnings"].append(f"Dados ausentes significativos ({missing_percentage:.1f}% do total)")

            # Usar validador principal para verificações mais detalhadas
            # Temporariamente comentado para evitar erro de dict unhashable
            # detailed_validation = self.data_validator.validate_dataframe(df, "streamlit_upload")
            # if not detailed_validation["is_valid"]:
            #     result["is_valid"] = False
            #     result["issues"].extend(detailed_validation["errors"])
            #
            # result["warnings"].extend(detailed_validation["warnings"])

            return result

        except Exception as e:
            self.logger.error(f"Erro na validação do CSV: {str(e)}")
            return {
                "is_valid": False,
                "issues": [f"Erro na validação do arquivo: {str(e)}"],
                "warnings": [],
                "info": {}
            }

    def validate_csv_for_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida se o CSV é adequado para análises EDA.

        Args:
            df: DataFrame com os dados

        Returns:
            Resultado da validação com recomendações para EDA
        """
        result = self.validate_csv_data(df)

        if not result["is_valid"]:
            return result

        # Adicionar informações específicas para EDA
        eda_info = {
            "suitable_for_descriptive": True,
            "suitable_for_correlation": False,
            "suitable_for_outlier_detection": False,
            "suitable_for_pattern_analysis": False,
            "recommended_analyses": []
        }

        # Verificar adequação para diferentes tipos de análise
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Análise descritiva sempre possível
        eda_info["recommended_analyses"].append("Análise Descritiva")

        # Correlação precisa de pelo menos 2 colunas numéricas
        if len(numeric_cols) >= 2:
            eda_info["suitable_for_correlation"] = True
            eda_info["recommended_analyses"].append("Análise de Correlação")

        # Detecção de outliers precisa de colunas numéricas
        if len(numeric_cols) > 0:
            eda_info["suitable_for_outlier_detection"] = True
            eda_info["recommended_analyses"].append("Detecção de Outliers")

        # Análise de padrões funciona com qualquer tipo de dados
        if len(df.columns) > 0:
            eda_info["suitable_for_pattern_analysis"] = True
            eda_info["recommended_analyses"].append("Análise de Padrões")

        result["eda_info"] = eda_info
        return result