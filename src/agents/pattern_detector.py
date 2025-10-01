"""
Agente especializado na detecção de padrões e tendências em dados.
Identifica sazonalidade, ciclicidade, valores frequentes e agrupamentos naturais,
fornecendo insights sobre comportamentos recorrentes nos datasets.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

from ..core.config import get_config
from ..core.llm_manager import get_llm_manager
from ..core.logger import get_logger, log_analysis_step, log_error_with_context
from ..models.analysis_result import PatternAnalysisResult, VisualizationResult
from ..models.enums import EDAAnalysisType, ProcessingStatus, VisualizationType


class PatternDetectionError(Exception):
    """Erro específico para detecção de padrões."""
    pass


class PatternDetectorAgent:
    """Agente especializado em detecção de padrões e tendências."""

    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        self.logger = get_logger("pattern_detector_agent")

    def analyze(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> PatternAnalysisResult:
        """
        Executa análise de padrões e tendências nos dados.

        Args:
            data: DataFrame para análise
            context: Contexto adicional da consulta

        Returns:
            PatternAnalysisResult com análise completa
        """
        start_time = time.time()

        try:
            log_analysis_step("pattern_detection", "started", {"shape": data.shape})

            result = PatternAnalysisResult(
                analysis_type=EDAAnalysisType.PATTERN,
                status=ProcessingStatus.IN_PROGRESS
            )

            # 1. Análise de padrões temporais
            result.temporal_patterns = self._analyze_temporal_patterns(data)

            # 2. Análise de frequências
            result.frequency_analysis = self._analyze_frequency_patterns(data)

            # 3. Identificação de agrupamentos
            result.clustering_results = self._identify_clusters(data)

            # 4. Identificação de tendências
            result.trends = self._identify_trends(data)

            # 5. Gerar insights usando LLM
            insights = self._generate_insights(data, result, context)
            result.insights = insights

            # 6. Marcar como concluído
            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - start_time

            log_analysis_step(
                "pattern_detection", "completed",
                {
                    "processing_time": result.processing_time,
                    "insights_count": len(result.insights),
                    "temporal_patterns_found": len(result.temporal_patterns),
                    "trends_identified": len(result.trends)
                }
            )

            return result

        except Exception as e:
            log_error_with_context(e, {"operation": "pattern_detection", "data_shape": data.shape})
            raise PatternDetectionError(f"Erro na detecção de padrões: {e}")

    def _analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões temporais nos dados."""
        try:
            temporal_analysis = {
                "datetime_columns": [],
                "temporal_patterns": {},
                "seasonality_detected": False,
                "cyclical_patterns": []
            }

            # Identificar colunas de data/hora
            datetime_columns = []
            for column in data.columns:
                col_data = data[column]

                # Verificar se é datetime
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    datetime_columns.append(column)
                elif pd.api.types.is_object_dtype(col_data):
                    # Tentar converter para datetime
                    try:
                        pd.to_datetime(col_data.dropna().head(100))
                        datetime_columns.append(column)
                    except (ValueError, TypeError):
                        pass

            temporal_analysis["datetime_columns"] = datetime_columns

            # Análise para cada coluna temporal
            for dt_col in datetime_columns[:3]:  # Limitar a 3 colunas
                try:
                    col_patterns = self._analyze_single_temporal_column(data, dt_col)
                    temporal_analysis["temporal_patterns"][dt_col] = col_patterns
                except Exception as e:
                    self.logger.warning(f"Erro ao analisar coluna temporal {dt_col}: {e}")

            # Verificar se há padrões baseados em índice numérico
            if not datetime_columns and len(data) > 10:
                temporal_analysis["index_patterns"] = self._analyze_index_patterns(data)

            return temporal_analysis

        except Exception as e:
            self.logger.error(f"Erro na análise temporal: {e}")
            return {"error": str(e)}

    def _analyze_single_temporal_column(self, data: pd.DataFrame, dt_col: str) -> Dict[str, Any]:
        """Analisa padrões em uma coluna temporal específica."""
        patterns = {
            "range": {},
            "frequency": {},
            "patterns": []
        }

        try:
            # Converter para datetime se necessário
            dt_series = pd.to_datetime(data[dt_col].dropna())

            if len(dt_series) < 2:
                return patterns

            # Análise de intervalo temporal
            patterns["range"] = {
                "start_date": dt_series.min().isoformat() if not dt_series.empty else None,
                "end_date": dt_series.max().isoformat() if not dt_series.empty else None,
                "duration_days": (dt_series.max() - dt_series.min()).days if len(dt_series) > 1 else 0
            }

            # Análise de frequência
            if len(dt_series) > 5:
                # Diferenças entre datas consecutivas
                time_diffs = dt_series.sort_values().diff().dropna()
                if not time_diffs.empty:
                    common_intervals = time_diffs.mode()
                    if not common_intervals.empty:
                        patterns["frequency"]["common_interval_days"] = common_intervals.iloc[0].days

                # Padrões por componentes de tempo
                dt_df = pd.DataFrame({
                    'datetime': dt_series,
                    'year': dt_series.dt.year,
                    'month': dt_series.dt.month,
                    'day_of_week': dt_series.dt.dayofweek,
                    'hour': dt_series.dt.hour
                })

                # Distribuição por dia da semana
                dow_counts = dt_df['day_of_week'].value_counts()
                patterns["frequency"]["day_of_week_distribution"] = dow_counts.to_dict()

                # Distribuição por mês
                month_counts = dt_df['month'].value_counts()
                patterns["frequency"]["month_distribution"] = month_counts.to_dict()

                # Identificar padrões
                if dow_counts.std() > dow_counts.mean() * 0.5:
                    patterns["patterns"].append("Variação significativa por dia da semana")

                if month_counts.std() > month_counts.mean() * 0.5:
                    patterns["patterns"].append("Variação sazonal detectada")

        except Exception as e:
            self.logger.warning(f"Erro ao analisar coluna temporal {dt_col}: {e}")

        return patterns

    def _analyze_index_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões baseados no índice dos dados."""
        index_patterns = {
            "sequential": False,
            "gaps_detected": False,
            "pattern_type": "unknown"
        }

        try:
            # Verificar se índice é sequencial
            if isinstance(data.index, pd.RangeIndex):
                index_patterns["sequential"] = True
                index_patterns["pattern_type"] = "sequential"
            elif data.index.is_monotonic_increasing:
                index_patterns["pattern_type"] = "monotonic_increasing"
            elif data.index.is_monotonic_decreasing:
                index_patterns["pattern_type"] = "monotonic_decreasing"

            # Detectar lacunas em dados numéricos sequenciais
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(data) > 10:
                # Analisar primeira coluna numérica para padrões
                first_col = data[numeric_cols[0]].dropna()
                if len(first_col) > 5:
                    # Detectar tendência
                    x = np.arange(len(first_col))
                    correlation = np.corrcoef(x, first_col)[0, 1]

                    if abs(correlation) > 0.7:
                        if correlation > 0:
                            index_patterns["trend"] = "increasing"
                        else:
                            index_patterns["trend"] = "decreasing"
                    else:
                        index_patterns["trend"] = "stable"

        except Exception as e:
            self.logger.warning(f"Erro ao analisar padrões de índice: {e}")

        return index_patterns

    def _analyze_frequency_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões de frequência nos dados."""
        try:
            frequency_analysis = {
                "categorical_patterns": {},
                "numerical_patterns": {},
                "top_values_per_column": {}
            }

            # Análise de colunas categóricas
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols[:5]:  # Limitar a 5 colunas
                try:
                    value_counts = data[col].value_counts()
                    if not value_counts.empty:
                        frequency_analysis["categorical_patterns"][col] = {
                            "most_frequent": str(value_counts.index[0]),
                            "most_frequent_count": int(value_counts.iloc[0]),
                            "unique_values": int(data[col].nunique()),
                            "top_3_values": value_counts.head(3).to_dict()
                        }

                        # Detectar padrões de distribuição
                        total_values = value_counts.sum()
                        top_percentage = (value_counts.iloc[0] / total_values) * 100

                        if top_percentage > 80:
                            frequency_analysis["categorical_patterns"][col]["pattern"] = "highly_concentrated"
                        elif top_percentage < 10:
                            frequency_analysis["categorical_patterns"][col]["pattern"] = "well_distributed"
                        else:
                            frequency_analysis["categorical_patterns"][col]["pattern"] = "moderately_concentrated"

                except Exception as e:
                    self.logger.warning(f"Erro ao analisar coluna categórica {col}: {e}")

            # Análise de colunas numéricas
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # Limitar a 5 colunas
                try:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Análise de distribuição de valores
                        value_counts = col_data.value_counts()

                        frequency_analysis["numerical_patterns"][col] = {
                            "unique_values": int(col_data.nunique()),
                            "most_frequent_value": float(value_counts.index[0]) if not value_counts.empty else None,
                            "most_frequent_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                            "repetition_ratio": len(col_data) / col_data.nunique() if col_data.nunique() > 0 else 0
                        }

                        # Detectar se valores são principalmente inteiros
                        if col_data.dtype in ['float64', 'float32']:
                            integer_ratio = (col_data % 1 == 0).sum() / len(col_data)
                            frequency_analysis["numerical_patterns"][col]["likely_integer"] = integer_ratio > 0.9

                except Exception as e:
                    self.logger.warning(f"Erro ao analisar coluna numérica {col}: {e}")

            return frequency_analysis

        except Exception as e:
            self.logger.error(f"Erro na análise de frequências: {e}")
            return {"error": str(e)}

    def _identify_clusters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identifica potenciais agrupamentos nos dados."""
        try:
            clustering_analysis = {
                "numeric_clustering": {},
                "categorical_clustering": {},
                "potential_groups": []
            }

            # Análise de agrupamentos em dados numéricos
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Análise básica de agrupamentos usando estatísticas
                numeric_data = data[numeric_cols].dropna()

                if len(numeric_data) > 10:
                    # Identificar potenciais clusters baseados em quartis
                    for col in numeric_cols[:3]:  # Limitar a 3 colunas
                        try:
                            col_data = numeric_data[col]
                            q1, q2, q3 = col_data.quantile([0.25, 0.5, 0.75])

                            clustering_analysis["numeric_clustering"][col] = {
                                "q1": float(q1),
                                "median": float(q2),
                                "q3": float(q3),
                                "iqr": float(q3 - q1),
                                "potential_clusters": self._identify_numeric_clusters(col_data)
                            }
                        except Exception as e:
                            self.logger.warning(f"Erro ao analisar clusters na coluna {col}: {e}")

            # Análise de agrupamentos categóricos
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols[:3]:  # Limitar a 3 colunas
                try:
                    value_counts = data[col].value_counts()
                    if len(value_counts) > 1:
                        clustering_analysis["categorical_clustering"][col] = {
                            "total_categories": len(value_counts),
                            "dominant_categories": value_counts.head(3).to_dict(),
                            "long_tail_categories": len(value_counts[value_counts == 1])
                        }
                except Exception as e:
                    self.logger.warning(f"Erro ao analisar clusters categóricos na coluna {col}: {e}")

            return clustering_analysis

        except Exception as e:
            self.logger.error(f"Erro na identificação de clusters: {e}")
            return {"error": str(e)}

    def _identify_numeric_clusters(self, col_data: pd.Series) -> List[Dict[str, Any]]:
        """Identifica clusters em dados numéricos usando métodos simples."""
        clusters = []

        try:
            # Método simples baseado em histograma
            hist, bin_edges = np.histogram(col_data, bins=min(10, len(col_data) // 10 + 1))

            for i, count in enumerate(hist):
                if count > len(col_data) * 0.1:  # Clusters com mais de 10% dos dados
                    clusters.append({
                        "range_start": float(bin_edges[i]),
                        "range_end": float(bin_edges[i + 1]),
                        "count": int(count),
                        "percentage": round((count / len(col_data)) * 100, 2)
                    })

        except Exception as e:
            self.logger.warning(f"Erro ao identificar clusters numéricos: {e}")

        return clusters

    def _identify_trends(self, data: pd.DataFrame) -> List[str]:
        """Identifica tendências gerais nos dados."""
        trends = []

        try:
            # Análise de tendências em colunas numéricas
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            for col in numeric_cols[:5]:  # Limitar a 5 colunas
                try:
                    col_data = data[col].dropna()
                    if len(col_data) > 5:
                        # Calcular correlação com índice para detectar tendência
                        x = np.arange(len(col_data))
                        correlation = np.corrcoef(x, col_data)[0, 1]

                        if abs(correlation) > 0.5:
                            if correlation > 0:
                                trends.append(f"Tendência de crescimento detectada na coluna '{col}'")
                            else:
                                trends.append(f"Tendência de decrescimento detectada na coluna '{col}'")

                        # Análise de variabilidade
                        cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                        if cv > 1:
                            trends.append(f"Alta variabilidade detectada na coluna '{col}' (CV={cv:.2f})")

                except Exception as e:
                    self.logger.warning(f"Erro ao analisar tendências na coluna {col}: {e}")

            # Análise de tendências categóricas
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols[:3]:  # Limitar a 3 colunas
                try:
                    value_counts = data[col].value_counts()
                    if len(value_counts) > 1:
                        # Detectar dominância de categorias
                        top_percentage = (value_counts.iloc[0] / value_counts.sum()) * 100
                        if top_percentage > 70:
                            trends.append(f"Forte concentração na categoria '{value_counts.index[0]}' da coluna '{col}' ({top_percentage:.1f}%)")

                except Exception as e:
                    self.logger.warning(f"Erro ao analisar tendências categóricas na coluna {col}: {e}")

        except Exception as e:
            self.logger.error(f"Erro na identificação de tendências: {e}")

        return trends[:10]  # Limitar a 10 tendências

    def _generate_insights(self, data: pd.DataFrame, result: PatternAnalysisResult, context: Optional[Dict[str, Any]]) -> List[str]:
        """Gera insights usando LLM baseado na análise de padrões."""
        try:
            # Preparar resumo para o LLM
            patterns_summary = f"""
            Análise de padrões em dataset com {len(data)} linhas e {len(data.columns)} colunas.

            Padrões temporais encontrados: {len(result.temporal_patterns)}
            Tendências identificadas: {len(result.trends)}

            Principais descobertas:
            """

            # Adicionar tendências identificadas
            for trend in result.trends[:5]:
                patterns_summary += f"\n- {trend}"

            # Adicionar informações de clustering
            if result.clustering_results.get("numeric_clustering"):
                patterns_summary += f"\nAgrupamentos numéricos identificados em {len(result.clustering_results['numeric_clustering'])} colunas."

            # Prompt para o LLM
            system_prompt = """
            Você é um especialista em análise de padrões e tendências de dados.
            Analise os padrões detectados e gere insights práticos sobre:
            1. Significado dos padrões encontrados
            2. Implicações dos agrupamentos identificados
            3. Relevância das tendências temporais
            4. Sugestões para investigação adicional

            Forneça entre 3-5 insights concisos e práticos.
            """

            user_prompt = f"""
            Padrões detectados em dataset:
            {patterns_summary}

            {f"Contexto da consulta: {context.get('query_text', '')}" if context else ""}

            Gere insights sobre os padrões encontrados.
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
            valid_insights = [insight for insight in insights if len(insight) > 20 and len(insight) < 200]

            return valid_insights[:5] if valid_insights else ["Análise de padrões concluída."]

        except Exception as e:
            self.logger.warning(f"Erro ao gerar insights com LLM: {e}")
            # Fallback para insights básicos
            return self._generate_basic_insights(result)

    def _generate_basic_insights(self, result: PatternAnalysisResult) -> List[str]:
        """Gera insights básicos sem LLM como fallback."""
        insights = []

        # Insights sobre tendências
        if result.trends:
            insights.append(f"Identificadas {len(result.trends)} tendências significativas nos dados.")

        # Insights sobre padrões temporais
        if result.temporal_patterns:
            insights.append("Padrões temporais detectados no dataset.")

        # Insights sobre clustering
        if result.clustering_results.get("numeric_clustering"):
            insights.append("Agrupamentos naturais identificados nos dados numéricos.")

        if not insights:
            insights.append("Análise de padrões concluída sem anomalias detectadas.")

        return insights[:3]


# Instância singleton
_pattern_detector_agent: Optional[PatternDetectorAgent] = None


def get_pattern_detector_agent() -> PatternDetectorAgent:
    """Obtém instância singleton do PatternDetectorAgent."""
    global _pattern_detector_agent
    if _pattern_detector_agent is None:
        _pattern_detector_agent = PatternDetectorAgent()
    return _pattern_detector_agent