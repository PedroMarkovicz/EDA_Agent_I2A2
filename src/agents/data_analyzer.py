"""
Agente responsável pela análise descritiva e estatística básica de datasets CSV.
Identifica tipos de dados, calcula medidas de tendência central e variabilidade,
e gera resumos estruturados para caracterização inicial dos dados.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.config import get_config
from ..core.llm_manager import get_llm_manager
from ..core.logger import get_logger, log_analysis_step, log_error_with_context
from ..models.analysis_result import DescriptiveAnalysisResult, StatisticalSummary, VisualizationResult
from ..models.enums import EDAAnalysisType, ProcessingStatus, VisualizationType


class DataAnalysisError(Exception):
    """Erro específico para análise de dados."""
    pass


class DataAnalyzerAgent:
    """Agente especializado em análise descritiva e estatística básica."""

    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        self.logger = get_logger("data_analyzer_agent")

    def analyze(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> DescriptiveAnalysisResult:
        """
        Executa análise descritiva completa dos dados.

        Args:
            data: DataFrame para análise
            context: Contexto adicional da consulta

        Returns:
            DescriptiveAnalysisResult com análise completa
        """
        start_time = time.time()

        try:
            log_analysis_step("descriptive_analysis", "started", {"shape": data.shape})

            result = DescriptiveAnalysisResult(
                analysis_type=EDAAnalysisType.DESCRIPTIVE,
                status=ProcessingStatus.IN_PROGRESS
            )

            # 1. Visão geral do dataset
            result.dataset_overview = self._generate_dataset_overview(data)

            # 2. Análise de tipos de dados
            result.data_types_summary = self._analyze_data_types(data)

            # 3. Resumos estatísticos por coluna
            result.column_summaries = self._generate_column_summaries(data)

            # 4. Análise de dados ausentes
            result.missing_data_summary = self._analyze_missing_data(data)

            # 5. Gerar insights usando LLM
            insights = self._generate_insights(data, result, context)
            result.insights = insights

            # 6. Marcar como concluído
            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - start_time

            log_analysis_step(
                "descriptive_analysis", "completed",
                {
                    "processing_time": result.processing_time,
                    "insights_count": len(result.insights),
                    "columns_analyzed": len(result.column_summaries)
                }
            )

            return result

        except Exception as e:
            log_error_with_context(e, {"operation": "descriptive_analysis", "data_shape": data.shape})
            raise DataAnalysisError(f"Erro na análise descritiva: {e}")

    def _generate_dataset_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Gera visão geral do dataset."""
        try:
            overview = {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "total_cells": data.size,
                "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
                "analysis_timestamp": datetime.now().isoformat(),
            }

            # Análise básica de completude
            total_missing = data.isnull().sum().sum()
            overview["total_missing_values"] = int(total_missing)
            overview["missing_percentage"] = round((total_missing / data.size) * 100, 2)
            overview["completeness_percentage"] = round(100 - overview["missing_percentage"], 2)

            # Duplicatas
            overview["duplicate_rows"] = int(data.duplicated().sum())
            overview["duplicate_percentage"] = round((overview["duplicate_rows"] / len(data)) * 100, 2)

            return overview

        except Exception as e:
            self.logger.error(f"Erro ao gerar visão geral do dataset: {e}")
            return {"error": str(e)}

    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, int]:
        """Analisa tipos de dados nas colunas."""
        try:
            type_counts = {
                "numeric": 0,
                "categorical": 0,
                "datetime": 0,
                "boolean": 0,
                "mixed": 0
            }

            for column in data.columns:
                col_data = data[column]

                if pd.api.types.is_numeric_dtype(col_data):
                    type_counts["numeric"] += 1
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    type_counts["datetime"] += 1
                elif pd.api.types.is_bool_dtype(col_data):
                    type_counts["boolean"] += 1
                elif pd.api.types.is_object_dtype(col_data):
                    # Verificar se pode ser categórico
                    unique_ratio = col_data.nunique() / len(col_data.dropna())
                    if unique_ratio < 0.5:  # Menos de 50% de valores únicos
                        type_counts["categorical"] += 1
                    else:
                        type_counts["mixed"] += 1
                else:
                    type_counts["mixed"] += 1

            return type_counts

        except Exception as e:
            self.logger.error(f"Erro na análise de tipos de dados: {e}")
            return {"error": str(e)}

    def _generate_column_summaries(self, data: pd.DataFrame) -> List[StatisticalSummary]:
        """Gera resumos estatísticos para cada coluna."""
        summaries = []

        for column in data.columns:
            try:
                col_data = data[column]
                summary = StatisticalSummary(
                    column_name=column,
                    data_type=self._classify_column_type(col_data),
                    count=int(col_data.count()),
                    missing_count=int(col_data.isnull().sum()),
                    missing_percentage=round((col_data.isnull().sum() / len(col_data)) * 100, 2),
                    unique_count=int(col_data.nunique())
                )

                # Estatísticas para dados numéricos
                if pd.api.types.is_numeric_dtype(col_data):
                    summary.mean = float(col_data.mean()) if not col_data.empty else None
                    summary.median = float(col_data.median()) if not col_data.empty else None
                    summary.std = float(col_data.std()) if not col_data.empty else None
                    summary.min_value = float(col_data.min()) if not col_data.empty else None
                    summary.max_value = float(col_data.max()) if not col_data.empty else None
                    summary.q25 = float(col_data.quantile(0.25)) if not col_data.empty else None
                    summary.q75 = float(col_data.quantile(0.75)) if not col_data.empty else None

                # Para dados categóricos e de texto
                if not col_data.empty:
                    value_counts = col_data.value_counts()
                    if not value_counts.empty:
                        summary.mode = value_counts.index[0]
                        summary.mode_frequency = int(value_counts.iloc[0])

                summaries.append(summary)

            except Exception as e:
                self.logger.warning(f"Erro ao processar coluna {column}: {e}")
                # Criar summary básico em caso de erro
                summaries.append(StatisticalSummary(
                    column_name=column,
                    data_type="error",
                    count=0,
                    missing_count=len(col_data),
                    missing_percentage=100.0,
                    unique_count=0
                ))

        return summaries

    def _classify_column_type(self, col_data: pd.Series) -> str:
        """Classifica o tipo de uma coluna específica."""
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                return "integer"
            else:
                return "float"
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            return "datetime"
        elif pd.api.types.is_bool_dtype(col_data):
            return "boolean"
        elif pd.api.types.is_object_dtype(col_data):
            # Determinar se é categórico ou texto
            unique_ratio = col_data.nunique() / len(col_data.dropna()) if len(col_data.dropna()) > 0 else 0
            if unique_ratio < 0.5:
                return "categorical"
            else:
                return "text"
        else:
            return "mixed"

    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões de dados ausentes."""
        try:
            missing_analysis = {
                "columns_with_missing": [],
                "missing_patterns": {},
                "completely_missing_columns": [],
                "mostly_complete_columns": []
            }

            for column in data.columns:
                missing_count = data[column].isnull().sum()
                missing_percentage = (missing_count / len(data)) * 100

                if missing_count > 0:
                    missing_info = {
                        "column": column,
                        "missing_count": int(missing_count),
                        "missing_percentage": round(missing_percentage, 2)
                    }
                    missing_analysis["columns_with_missing"].append(missing_info)

                    if missing_percentage == 100:
                        missing_analysis["completely_missing_columns"].append(column)
                    elif missing_percentage < 5:
                        missing_analysis["mostly_complete_columns"].append(column)

            # Análise de padrões de missing data
            total_missing = data.isnull().sum().sum()
            missing_analysis["total_missing_values"] = int(total_missing)
            missing_analysis["overall_missing_percentage"] = round((total_missing / data.size) * 100, 2)

            # Linhas com dados ausentes
            rows_with_missing = data.isnull().any(axis=1).sum()
            missing_analysis["rows_with_missing"] = int(rows_with_missing)
            missing_analysis["complete_rows"] = int(len(data) - rows_with_missing)

            return missing_analysis

        except Exception as e:
            self.logger.error(f"Erro na análise de dados ausentes: {e}")
            return {"error": str(e)}

    def _generate_insights(self, data: pd.DataFrame, result: DescriptiveAnalysisResult, context: Optional[Dict[str, Any]]) -> List[str]:
        """Gera insights usando LLM baseado na análise descritiva."""
        try:
            # Extrair query do contexto
            user_query = context.get('query', '') if context else ''

            # Preparar resumo para o LLM
            dataset_summary = f"""
            Dataset com {result.dataset_overview.get('total_rows', 0)} linhas e {result.dataset_overview.get('total_columns', 0)} colunas.
            Completude: {result.dataset_overview.get('completeness_percentage', 0):.1f}%
            Tipos de dados: {result.data_types_summary}

            Resumo das colunas:
            """

            for summary in result.column_summaries[:10]:  # Limitar a 10 colunas
                if summary.data_type in ["integer", "float"]:
                    dataset_summary += f"\n- {summary.column_name} ({summary.data_type}): μ={summary.mean:.2f}, σ={summary.std:.2f}, min={summary.min_value}, max={summary.max_value}" if summary.mean else f"\n- {summary.column_name} ({summary.data_type}): sem dados"
                else:
                    dataset_summary += f"\n- {summary.column_name} ({summary.data_type}): {summary.unique_count} valores únicos"

            # Prompt para o LLM
            system_prompt = """
            Você é um especialista em análise exploratória de dados.
            Sua tarefa é responder ESPECIFICAMENTE à pergunta do usuário com base nos dados analisados.

            IMPORTANTE:
            - Se o usuário perguntou sobre um valor específico (máximo, mínimo, média, etc.), responda DIRETAMENTE com esse valor
            - NÃO forneça apenas uma visão geral genérica do dataset
            - Priorize responder à pergunta exata do usuário
            - Seja específico e direto na resposta

            Forneça entre 1-3 insights RELEVANTES à pergunta do usuário.
            """

            user_prompt = f"""
            Pergunta do usuário: "{user_query}"

            Análise descritiva do dataset:
            {dataset_summary}

            Responda ESPECIFICAMENTE à pergunta do usuário usando os dados acima.
            Se o usuário perguntou sobre um valor específico de uma coluna, forneça esse valor diretamente.
            """

            response = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
            )

            # Processar resposta do LLM
            insights_text = response["content"]
            insights = [
                line.strip().lstrip("- ").lstrip("• ").lstrip("* ")
                for line in insights_text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            # Filtrar insights válidos
            valid_insights = [insight for insight in insights if len(insight) > 20 and len(insight) < 300]

            return valid_insights[:5] if valid_insights else self._generate_basic_insights(result, user_query)

        except Exception as e:
            self.logger.warning(f"Erro ao gerar insights com LLM: {e}")
            # Fallback para insights básicos
            return self._generate_basic_insights(result, user_query if 'user_query' in locals() else '')

    def _generate_basic_insights(self, result: DescriptiveAnalysisResult, user_query: str = '') -> List[str]:
        """Gera insights básicos sem LLM como fallback, tentando responder à query específica."""
        insights = []
        query_lower = user_query.lower()

        # Tentar responder queries específicas sobre colunas
        if any(keyword in query_lower for keyword in ['máximo', 'max', 'maximum', 'maior']):
            # Buscar coluna mencionada
            for summary in result.column_summaries:
                if summary.column_name.lower() in query_lower and summary.max_value is not None:
                    insights.append(f"O valor máximo da coluna '{summary.column_name}' é {summary.max_value}.")
                    break

        if any(keyword in query_lower for keyword in ['mínimo', 'min', 'minimum', 'menor']):
            # Buscar coluna mencionada
            for summary in result.column_summaries:
                if summary.column_name.lower() in query_lower and summary.min_value is not None:
                    insights.append(f"O valor mínimo da coluna '{summary.column_name}' é {summary.min_value}.")
                    break

        if any(keyword in query_lower for keyword in ['média', 'mean', 'average']):
            # Buscar coluna mencionada
            for summary in result.column_summaries:
                if summary.column_name.lower() in query_lower and summary.mean is not None:
                    insights.append(f"A média da coluna '{summary.column_name}' é {summary.mean:.2f}.")
                    break

        if any(keyword in query_lower for keyword in ['soma', 'sum', 'total']):
            # Buscar coluna mencionada
            for summary in result.column_summaries:
                if summary.column_name.lower() in query_lower and summary.mean is not None:
                    # Calcular soma aproximada (média * count)
                    total = summary.mean * summary.count
                    insights.append(f"A soma total da coluna '{summary.column_name}' é aproximadamente {total:.2f}.")
                    break

        # Se não encontrou resposta específica, dar insights gerais
        if not insights:
            overview = result.dataset_overview

            # Insight sobre completude
            completeness = overview.get("completeness_percentage", 0)
            if completeness > 95:
                insights.append("Dataset possui excelente qualidade com mais de 95% de completude.")
            elif completeness > 80:
                insights.append("Dataset possui boa qualidade com mais de 80% de completude.")
            else:
                insights.append(f"Dataset possui {completeness:.1f}% de completude, requer atenção aos dados ausentes.")

            # Insight sobre tipos de dados
            types = result.data_types_summary
            if types.get("numeric", 0) > types.get("categorical", 0):
                insights.append("Dataset é predominantemente numérico, adequado para análises estatísticas.")
            else:
                insights.append("Dataset contém muitas variáveis categóricas, ideal para análises de segmentação.")

            # Insight sobre duplicatas
            if overview.get("duplicate_percentage", 0) > 5:
                insights.append(f"Identificadas {overview.get('duplicate_percentage', 0):.1f}% de linhas duplicadas.")

        return insights[:3]


# Instância singleton
_data_analyzer_agent: Optional[DataAnalyzerAgent] = None


def get_data_analyzer_agent() -> DataAnalyzerAgent:
    """Obtém instância singleton do DataAnalyzerAgent."""
    global _data_analyzer_agent
    if _data_analyzer_agent is None:
        _data_analyzer_agent = DataAnalyzerAgent()
    return _data_analyzer_agent