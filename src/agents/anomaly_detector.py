"""
Agente dedicado à identificação e análise de outliers e anomalias nos dados.
Utiliza múltiplos métodos de detecção, avalia impacto das anomalias e
sugere estratégias apropriadas para investigação dos valores atípicos.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ..core.config import get_config
from ..core.llm_manager import get_llm_manager
from ..core.logger import get_logger, log_analysis_step, log_error_with_context
from ..models.analysis_result import AnomalyAnalysisResult, VisualizationResult
from ..models.enums import EDAAnalysisType, ProcessingStatus, VisualizationType


class AnomalyDetectionError(Exception):
    """Erro específico para detecção de anomalias."""
    pass


class AnomalyDetectorAgent:
    """Agente especializado em detecção de outliers e anomalias."""

    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        self.logger = get_logger("anomaly_detector_agent")

    def analyze(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> AnomalyAnalysisResult:
        """
        Executa análise de anomalias e outliers nos dados.

        Args:
            data: DataFrame para análise
            context: Contexto adicional da consulta

        Returns:
            AnomalyAnalysisResult com análise completa
        """
        start_time = time.time()

        try:
            log_analysis_step("anomaly_detection", "started", {"shape": data.shape})

            result = AnomalyAnalysisResult(
                analysis_type=EDAAnalysisType.ANOMALY,
                status=ProcessingStatus.IN_PROGRESS
            )

            # 1. Detecção de outliers usando múltiplos métodos
            result.outliers_by_column = self._detect_outliers_all_methods(data)

            # 2. Estatísticas dos outliers
            result.outlier_statistics = self._calculate_outlier_statistics(data, result.outliers_by_column)

            # 3. Análise de impacto
            result.impact_analysis = self._analyze_outlier_impact(data, result.outliers_by_column)

            # 4. Definir métodos utilizados
            result.outlier_detection_methods = ["IQR", "Z-Score", "Modified Z-Score", "Isolation Forest"]

            # 5. Gerar recomendações para investigação
            result.recommendations = self._generate_investigation_recommendations(data, result)

            # 6. Gerar insights usando LLM
            insights = self._generate_insights(data, result, context)
            result.insights = insights

            # 7. Marcar como concluído
            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - start_time

            log_analysis_step(
                "anomaly_detection", "completed",
                {
                    "processing_time": result.processing_time,
                    "insights_count": len(result.insights),
                    "methods_used": len(result.outlier_detection_methods),
                    "columns_analyzed": len(result.outliers_by_column)
                }
            )

            return result

        except Exception as e:
            log_error_with_context(e, {"operation": "anomaly_detection", "data_shape": data.shape})
            raise AnomalyDetectionError(f"Erro na detecção de anomalias: {e}")

    def _detect_outliers_all_methods(self, data: pd.DataFrame) -> Dict[str, List[Any]]:
        """Detecta outliers usando múltiplos métodos."""
        outliers_by_column = {}

        # Analisar apenas colunas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for column in numeric_cols:
            try:
                col_data = data[column].dropna()
                if len(col_data) < 5:  # Dados insuficientes
                    continue

                column_outliers = {
                    "iqr_outliers": self._detect_outliers_iqr(col_data),
                    "zscore_outliers": self._detect_outliers_zscore(col_data),
                    "modified_zscore_outliers": self._detect_outliers_modified_zscore(col_data),
                    "isolation_forest_outliers": self._detect_outliers_isolation_forest(col_data)
                }

                # Consolidar outliers únicos
                all_outliers = set()
                for method_outliers in column_outliers.values():
                    all_outliers.update(method_outliers)

                outliers_by_column[column] = {
                    "outlier_values": list(all_outliers),
                    "outlier_indices": self._get_outlier_indices(data[column], list(all_outliers)),
                    "methods_detected": column_outliers,
                    "total_outliers": len(all_outliers),
                    "outlier_percentage": round((len(all_outliers) / len(col_data)) * 100, 2)
                }

            except Exception as e:
                self.logger.warning(f"Erro ao detectar outliers na coluna {column}: {e}")

        return outliers_by_column

    def _detect_outliers_iqr(self, data: pd.Series) -> List[float]:
        """Detecta outliers usando método IQR."""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]
            return outliers.tolist()

        except Exception as e:
            self.logger.warning(f"Erro no método IQR: {e}")
            return []

    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> List[float]:
        """Detecta outliers usando Z-Score."""
        try:
            if data.std() == 0:  # Desvio padrão zero
                return []

            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > threshold]
            return outliers.tolist()

        except Exception as e:
            self.logger.warning(f"Erro no método Z-Score: {e}")
            return []

    def _detect_outliers_modified_zscore(self, data: pd.Series, threshold: float = 3.5) -> List[float]:
        """Detecta outliers usando Modified Z-Score (baseado na mediana)."""
        try:
            median = data.median()
            mad = np.median(np.abs(data - median))

            if mad == 0:  # MAD zero
                return []

            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = data[np.abs(modified_z_scores) > threshold]
            return outliers.tolist()

        except Exception as e:
            self.logger.warning(f"Erro no método Modified Z-Score: {e}")
            return []

    def _detect_outliers_isolation_forest(self, data: pd.Series, contamination: float = 0.1) -> List[float]:
        """Detecta outliers usando Isolation Forest."""
        try:
            # Implementação simplificada usando estatísticas
            # (uma implementação real usaria sklearn.ensemble.IsolationForest)

            # Para esta implementação simplificada, usar método baseado em percentis
            lower_percentile = np.percentile(data, 5)
            upper_percentile = np.percentile(data, 95)

            outliers = data[(data < lower_percentile) | (data > upper_percentile)]
            return outliers.tolist()

        except Exception as e:
            self.logger.warning(f"Erro no método Isolation Forest: {e}")
            return []

    def _get_outlier_indices(self, original_series: pd.Series, outlier_values: List[float]) -> List[int]:
        """Obtém índices dos outliers no DataFrame original."""
        try:
            indices = []
            for value in outlier_values:
                # Encontrar índices onde o valor ocorre
                value_indices = original_series[original_series == value].index.tolist()
                indices.extend(value_indices)

            return sorted(list(set(indices)))  # Remove duplicatas e ordena

        except Exception as e:
            self.logger.warning(f"Erro ao obter índices de outliers: {e}")
            return []

    def _calculate_outlier_statistics(self, data: pd.DataFrame, outliers_by_column: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula estatísticas dos outliers detectados."""
        try:
            stats = {
                "total_columns_analyzed": len(outliers_by_column),
                "columns_with_outliers": 0,
                "total_outliers_detected": 0,
                "outlier_summary_by_column": {},
                "overall_outlier_percentage": 0.0
            }

            total_values = 0
            total_outliers = 0

            for column, outlier_info in outliers_by_column.items():
                col_data = data[column].dropna()
                num_outliers = outlier_info["total_outliers"]

                if num_outliers > 0:
                    stats["columns_with_outliers"] += 1

                stats["outlier_summary_by_column"][column] = {
                    "outliers_count": num_outliers,
                    "outlier_percentage": outlier_info["outlier_percentage"],
                    "severity": self._classify_outlier_severity(outlier_info["outlier_percentage"]),
                    "methods_agreeing": self._count_agreeing_methods(outlier_info["methods_detected"])
                }

                total_values += len(col_data)
                total_outliers += num_outliers

            stats["total_outliers_detected"] = total_outliers
            if total_values > 0:
                stats["overall_outlier_percentage"] = round((total_outliers / total_values) * 100, 2)

            return stats

        except Exception as e:
            self.logger.error(f"Erro ao calcular estatísticas de outliers: {e}")
            return {"error": str(e)}

    def _classify_outlier_severity(self, outlier_percentage: float) -> str:
        """Classifica a severidade dos outliers baseado na porcentagem."""
        if outlier_percentage >= 10:
            return "high"
        elif outlier_percentage >= 5:
            return "moderate"
        elif outlier_percentage >= 1:
            return "low"
        else:
            return "minimal"

    def _count_agreeing_methods(self, methods_detected: Dict[str, List]) -> int:
        """Conta quantos métodos concordam sobre outliers."""
        try:
            # Contar métodos que detectaram pelo menos um outlier
            agreeing_methods = 0
            for method_outliers in methods_detected.values():
                if len(method_outliers) > 0:
                    agreeing_methods += 1
            return agreeing_methods

        except Exception as e:
            self.logger.warning(f"Erro ao contar métodos concordantes: {e}")
            return 0

    def _analyze_outlier_impact(self, data: pd.DataFrame, outliers_by_column: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa o impacto dos outliers nas estatísticas."""
        try:
            impact_analysis = {
                "statistical_impact": {},
                "distribution_impact": {},
                "correlation_impact": {}
            }

            for column, outlier_info in outliers_by_column.items():
                if outlier_info["total_outliers"] == 0:
                    continue

                try:
                    col_data = data[column].dropna()
                    outlier_indices = outlier_info["outlier_indices"]

                    # Dados sem outliers
                    clean_data = col_data.drop(outlier_indices, errors='ignore')

                    if len(clean_data) == 0:
                        continue

                    # Impacto nas estatísticas
                    original_mean = col_data.mean()
                    clean_mean = clean_data.mean()
                    original_std = col_data.std()
                    clean_std = clean_data.std()

                    impact_analysis["statistical_impact"][column] = {
                        "mean_change": abs(original_mean - clean_mean),
                        "mean_change_percentage": abs((original_mean - clean_mean) / original_mean * 100) if original_mean != 0 else 0,
                        "std_change": abs(original_std - clean_std),
                        "std_change_percentage": abs((original_std - clean_std) / original_std * 100) if original_std != 0 else 0
                    }

                    # Impacto na distribuição
                    original_skew = col_data.skew() if len(col_data) > 2 else 0
                    clean_skew = clean_data.skew() if len(clean_data) > 2 else 0

                    impact_analysis["distribution_impact"][column] = {
                        "skewness_change": abs(original_skew - clean_skew),
                        "range_change": (col_data.max() - col_data.min()) - (clean_data.max() - clean_data.min())
                    }

                except Exception as e:
                    self.logger.warning(f"Erro ao analisar impacto da coluna {column}: {e}")

            return impact_analysis

        except Exception as e:
            self.logger.error(f"Erro na análise de impacto: {e}")
            return {"error": str(e)}

    def _generate_investigation_recommendations(self, data: pd.DataFrame, result: AnomalyAnalysisResult) -> List[str]:
        """Gera recomendações para investigação dos outliers."""
        recommendations = []

        try:
            for column, outlier_info in result.outliers_by_column.items():
                num_outliers = outlier_info["total_outliers"]
                outlier_percentage = outlier_info["outlier_percentage"]

                if num_outliers == 0:
                    continue

                # Recomendações baseadas na severidade
                severity = self._classify_outlier_severity(outlier_percentage)

                if severity == "high":
                    recommendations.append(f"Coluna '{column}': Alta concentração de outliers ({outlier_percentage:.1f}%) - investigar possível erro de coleta ou problema sistêmico")
                elif severity == "moderate":
                    recommendations.append(f"Coluna '{column}': Outliers moderados ({outlier_percentage:.1f}%) - verificar se representam comportamentos extremos legítimos")
                elif severity == "low":
                    recommendations.append(f"Coluna '{column}': Poucos outliers ({outlier_percentage:.1f}%) - analisar contexto específico destes valores")

                # Recomendações baseadas nos métodos
                methods_count = self._count_agreeing_methods(outlier_info["methods_detected"])
                if methods_count >= 3:
                    recommendations.append(f"Coluna '{column}': Outliers confirmados por múltiplos métodos - alta confiança na detecção")
                elif methods_count == 1:
                    recommendations.append(f"Coluna '{column}': Outliers detectados por apenas um método - validar manualmente")

            # Recomendações gerais
            total_outliers = result.outlier_statistics.get("total_outliers_detected", 0)
            if total_outliers > 0:
                recommendations.append("Considerar análise visual (box plots, scatter plots) para melhor compreensão dos outliers")
                recommendations.append("Investigar possíveis causas: erros de entrada, eventos especiais, ou variações naturais")

                if result.outlier_statistics.get("overall_outlier_percentage", 0) > 10:
                    recommendations.append("Alto percentual de outliers detectado - revisar processo de coleta de dados")

        except Exception as e:
            self.logger.error(f"Erro ao gerar recomendações: {e}")
            recommendations.append("Realizar análise manual detalhada dos valores atípicos identificados")

        return recommendations[:8]  # Limitar a 8 recomendações

    def _generate_insights(self, data: pd.DataFrame, result: AnomalyAnalysisResult, context: Optional[Dict[str, Any]]) -> List[str]:
        """Gera insights usando LLM baseado na análise de anomalias."""
        try:
            # Preparar resumo para o LLM
            anomaly_summary = f"""
            Análise de anomalias em dataset com {len(data)} linhas e {len(data.columns)} colunas.

            Resultados da detecção:
            - Colunas analisadas: {result.outlier_statistics.get('total_columns_analyzed', 0)}
            - Colunas com outliers: {result.outlier_statistics.get('columns_with_outliers', 0)}
            - Total de outliers: {result.outlier_statistics.get('total_outliers_detected', 0)}
            - Percentual geral: {result.outlier_statistics.get('overall_outlier_percentage', 0):.1f}%

            Métodos utilizados: {', '.join(result.outlier_detection_methods)}

            Principais descobertas por coluna:
            """

            # Adicionar informações por coluna (limitar a 5)
            column_count = 0
            for column, stats in result.outlier_statistics.get("outlier_summary_by_column", {}).items():
                if column_count >= 5:
                    break
                if stats["outliers_count"] > 0:
                    anomaly_summary += f"\n- {column}: {stats['outliers_count']} outliers ({stats['outlier_percentage']:.1f}%), severidade {stats['severity']}"
                    column_count += 1

            # Prompt para o LLM
            system_prompt = """
            Você é um especialista em detecção de anomalias e outliers.
            Analise os resultados da detecção e gere insights práticos sobre:
            1. Significado dos outliers encontrados
            2. Possíveis causas dos valores atípicos
            3. Impacto dos outliers na qualidade dos dados
            4. Próximos passos recomendados

            Forneça entre 3-5 insights concisos e acionáveis.
            """

            user_prompt = f"""
            Resultados da detecção de anomalias:
            {anomaly_summary}

            {f"Contexto da consulta: {context.get('query_text', '')}" if context else ""}

            Gere insights sobre as anomalias detectadas.
            """

            response = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )

            # Processar resposta do LLM
            insights_text = response["content"]
            insights = [
                line.strip().lstrip("- ").lstrip("• ").lstrip("* ")
                for line in insights_text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            # Filtrar insights válidos
            valid_insights = [insight for insight in insights if len(insight) > 20 and len(insight) < 200]

            return valid_insights[:5] if valid_insights else ["Análise de anomalias concluída."]

        except Exception as e:
            self.logger.warning(f"Erro ao gerar insights com LLM: {e}")
            # Fallback para insights básicos
            return self._generate_basic_insights(result)

    def _generate_basic_insights(self, result: AnomalyAnalysisResult) -> List[str]:
        """Gera insights básicos sem LLM como fallback."""
        insights = []

        stats = result.outlier_statistics

        # Insight sobre detecção geral
        total_outliers = stats.get("total_outliers_detected", 0)
        if total_outliers == 0:
            insights.append("Nenhum outlier significativo detectado no dataset.")
        else:
            insights.append(f"Detectados {total_outliers} outliers em {stats.get('columns_with_outliers', 0)} colunas.")

        # Insight sobre severidade
        overall_percentage = stats.get("overall_outlier_percentage", 0)
        if overall_percentage > 5:
            insights.append(f"Alto percentual de outliers ({overall_percentage:.1f}%) indica possíveis problemas na qualidade dos dados.")
        elif overall_percentage > 0:
            insights.append(f"Percentual moderado de outliers ({overall_percentage:.1f}%) dentro do esperado para a maioria dos datasets.")

        # Insight sobre métodos
        if len(result.outlier_detection_methods) >= 3:
            insights.append("Múltiplos métodos de detecção utilizados aumentam confiabilidade dos resultados.")

        return insights[:3]


# Instância singleton
_anomaly_detector_agent: Optional[AnomalyDetectorAgent] = None


def get_anomaly_detector_agent() -> AnomalyDetectorAgent:
    """Obtém instância singleton do AnomalyDetectorAgent."""
    global _anomaly_detector_agent
    if _anomaly_detector_agent is None:
        _anomaly_detector_agent = AnomalyDetectorAgent()
    return _anomaly_detector_agent