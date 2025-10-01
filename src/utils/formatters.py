"""
Formatadores para respostas textuais e apresentacao de resultados EDA.
Converte dados de analise em formato legivel e estruturado para exibicao.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from ..core.config import get_config
from ..core.logger import get_logger, log_data_operation, log_error_with_context
from ..models.enums import EDAAnalysisType, VisualizationType
from ..models.analysis_result import (
    DescriptiveAnalysisResult,
    PatternAnalysisResult,
    AnomalyAnalysisResult,
    RelationshipAnalysisResult,
    AnalysisConclusion
)


class FormatterError(Exception):
    """Erro especifico do formatador."""
    pass


class EDAResponseFormatter:
    """Formatador de respostas para analise exploratoria de dados."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("response_formatter")

    def format_descriptive_analysis(self, result: DescriptiveAnalysisResult) -> str:
        """
        Formata resultado de analise descritiva em texto legivel.

        Args:
            result: Resultado da analise descritiva

        Returns:
            Texto formatado
        """
        try:
            formatted_text = []

            # Cabecalho
            formatted_text.append("# ANALISE DESCRITIVA DO DATASET")
            formatted_text.append("=" * 50)
            formatted_text.append("")

            # Visao geral do dataset
            if result.dataset_overview:
                formatted_text.append("## Visao Geral")
                overview = result.dataset_overview
                formatted_text.append(f"- Total de registros: {self._format_number(overview.get('total_rows', 0))}")
                formatted_text.append(f"- Total de colunas: {overview.get('total_columns', 0)}")
                formatted_text.append(f"- Memoria utilizada: {overview.get('memory_usage_mb', 0):.1f} MB")

                if overview.get('missing_data_percentage', 0) > 0:
                    formatted_text.append(f"- Dados ausentes: {overview.get('missing_data_percentage', 0):.1f}%")

                formatted_text.append("")

            # Resumo dos tipos de dados
            if result.data_types_summary:
                formatted_text.append("## Tipos de Dados")
                types_summary = result.data_types_summary
                for data_type, count in types_summary.items():
                    formatted_text.append(f"- {data_type.replace('_', ' ').title()}: {count} colunas")
                formatted_text.append("")

            # Resumos das colunas
            if result.column_summaries:
                formatted_text.append("## Resumo das Colunas")
                for summary in result.column_summaries:
                    formatted_text.append(f"### {summary.column_name}")
                    formatted_text.append(f"- Tipo: {summary.data_type}")
                    formatted_text.append(f"- Valores validos: {self._format_number(summary.count)}")
                    formatted_text.append(f"- Valores unicos: {self._format_number(summary.unique_count)}")

                    if summary.missing_count > 0:
                        formatted_text.append(f"- Valores ausentes: {self._format_number(summary.missing_count)} ({summary.missing_percentage:.1f}%)")

                    # Estatisticas para dados numericos
                    if summary.mean is not None:
                        formatted_text.append("- Estatisticas:")
                        formatted_text.append(f"  * Media: {self._format_number(summary.mean)}")
                        formatted_text.append(f"  * Mediana: {self._format_number(summary.median)}")
                        formatted_text.append(f"  * Desvio padrao: {self._format_number(summary.std)}")
                        formatted_text.append(f"  * Minimo: {self._format_number(summary.min_value)}")
                        formatted_text.append(f"  * Maximo: {self._format_number(summary.max_value)}")

                    # Valores mais frequentes para dados categoricos
                    if summary.mode is not None:
                        formatted_text.append("- Valor mais frequente:")
                        formatted_text.append(f"  * {summary.mode}: {self._format_number(summary.mode_frequency) if summary.mode_frequency else 'N/A'}")

                    formatted_text.append("")

            log_data_operation("descriptive_analysis_formatted", {"columns_count": len(result.column_summaries) if result.column_summaries else 0}, "response_formatter")

            return "\n".join(formatted_text)

        except Exception as e:
            log_error_with_context(e, {"result_type": type(result).__name__}, "descriptive_formatting_error")
            raise FormatterError(f"Erro ao formatar analise descritiva: {str(e)}")

    def format_pattern_analysis(self, result: PatternAnalysisResult) -> str:
        """
        Formata resultado de analise de padroes em texto legivel.

        Args:
            result: Resultado da analise de padroes

        Returns:
            Texto formatado
        """
        try:
            formatted_text = []

            formatted_text.append("# ANALISE DE PADROES E TENDENCIAS")
            formatted_text.append("=" * 50)
            formatted_text.append("")

            # Padroes temporais
            if result.temporal_patterns:
                formatted_text.append("## Padroes Temporais")
                for pattern in result.temporal_patterns:
                    formatted_text.append(f"- {pattern}")
                formatted_text.append("")

            # Padroes de frequencia
            if result.frequency_patterns:
                formatted_text.append("## Padroes de Frequencia")
                for column, patterns in result.frequency_patterns.items():
                    formatted_text.append(f"### Coluna: {column}")
                    for pattern in patterns:
                        formatted_text.append(f"- {pattern}")
                formatted_text.append("")

            # Clusters identificados
            if result.clusters:
                formatted_text.append("## Agrupamentos Identificados")
                for i, cluster in enumerate(result.clusters, 1):
                    formatted_text.append(f"### Cluster {i}")
                    formatted_text.append(f"- Tamanho: {cluster.get('size', 'N/A')}")
                    if cluster.get('characteristics'):
                        formatted_text.append("- Caracteristicas:")
                        for char in cluster['characteristics']:
                            formatted_text.append(f"  * {char}")
                formatted_text.append("")

            # Trends
            if result.trends:
                formatted_text.append("## Tendencias Detectadas")
                for trend in result.trends:
                    formatted_text.append(f"- {trend}")
                formatted_text.append("")

            if not any([result.temporal_patterns, result.frequency_patterns, result.clusters, result.trends]):
                formatted_text.append("Nenhum padrao significativo detectado nos dados.")

            return "\n".join(formatted_text)

        except Exception as e:
            log_error_with_context(e, {"result_type": type(result).__name__}, "pattern_formatting_error")
            raise FormatterError(f"Erro ao formatar analise de padroes: {str(e)}")

    def format_anomaly_analysis(self, result: AnomalyAnalysisResult) -> str:
        """
        Formata resultado de analise de anomalias em texto legivel.

        Args:
            result: Resultado da analise de anomalias

        Returns:
            Texto formatado
        """
        try:
            formatted_text = []

            formatted_text.append("# ANALISE DE ANOMALIAS E OUTLIERS")
            formatted_text.append("=" * 50)
            formatted_text.append("")

            if result.outliers_by_column:
                formatted_text.append("## Outliers por Coluna")
                for column, outliers_info in result.outliers_by_column.items():
                    formatted_text.append(f"### {column}")
                    formatted_text.append(f"- Total de outliers: {len(outliers_info.get('outliers', []))}")
                    formatted_text.append(f"- Porcentagem: {outliers_info.get('percentage', 0):.2f}%")

                    if outliers_info.get('method'):
                        formatted_text.append(f"- Metodo de deteccao: {outliers_info['method']}")

                    if outliers_info.get('bounds'):
                        bounds = outliers_info['bounds']
                        formatted_text.append(f"- Limite inferior: {self._format_number(bounds.get('lower', 'N/A'))}")
                        formatted_text.append(f"- Limite superior: {self._format_number(bounds.get('upper', 'N/A'))}")

                    # Mostrar alguns exemplos de outliers
                    outliers = outliers_info.get('outliers', [])
                    if outliers:
                        examples = outliers[:3]  # Primeiros 3 outliers
                        formatted_text.append("- Exemplos de outliers:")
                        for outlier in examples:
                            formatted_text.append(f"  * {self._format_number(outlier)}")
                        if len(outliers) > 3:
                            formatted_text.append(f"  * ... e mais {len(outliers) - 3} outliers")

                    formatted_text.append("")

            # Deteccoes por metodo
            if result.detection_methods:
                formatted_text.append("## Metodos de Deteccao Utilizados")
                for method, info in result.detection_methods.items():
                    formatted_text.append(f"- {method}: {info.get('description', 'Metodo de deteccao de outliers')}")
                formatted_text.append("")

            # Impacto das anomalias
            if result.impact_analysis:
                formatted_text.append("## Impacto das Anomalias")
                impact = result.impact_analysis
                if impact.get('statistical_impact'):
                    formatted_text.append("- Impacto estatistico:")
                    for stat, value in impact['statistical_impact'].items():
                        formatted_text.append(f"  * {stat}: {self._format_number(value)}")
                formatted_text.append("")

            # Recomendacoes
            if result.recommendations:
                formatted_text.append("## Recomendacoes")
                for rec in result.recommendations:
                    formatted_text.append(f"- {rec}")
                formatted_text.append("")

            if not result.outliers_by_column:
                formatted_text.append("Nenhuma anomalia significativa detectada nos dados.")

            return "\n".join(formatted_text)

        except Exception as e:
            log_error_with_context(e, {"result_type": type(result).__name__}, "anomaly_formatting_error")
            raise FormatterError(f"Erro ao formatar analise de anomalias: {str(e)}")

    def format_relationship_analysis(self, result: RelationshipAnalysisResult) -> str:
        """
        Formata resultado de analise de relacionamentos em texto legivel.

        Args:
            result: Resultado da analise de relacionamentos

        Returns:
            Texto formatado
        """
        try:
            formatted_text = []

            formatted_text.append("# ANALISE DE RELACIONAMENTOS E CORRELACOES")
            formatted_text.append("=" * 50)
            formatted_text.append("")

            # Matriz de correlacao
            if result.correlation_matrix:
                formatted_text.append("## Correlacoes Mais Significativas")

                # Converter matriz para pares ordenados
                correlations = []
                for var1, correlations_dict in result.correlation_matrix.items():
                    for var2, corr_value in correlations_dict.items():
                        if var1 != var2 and abs(corr_value) > 0.1:  # Apenas correlacoes significativas
                            correlations.append((var1, var2, corr_value))

                # Ordenar por valor absoluto de correlacao
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)

                for var1, var2, corr in correlations[:10]:  # Top 10
                    strength = self._correlation_strength(abs(corr))
                    direction = "positiva" if corr > 0 else "negativa"
                    formatted_text.append(f"- {var1} ↔ {var2}: {corr:.3f} ({strength} {direction})")

                if not correlations:
                    formatted_text.append("- Nenhuma correlacao significativa encontrada")
                formatted_text.append("")

            # Relacionamentos nao-lineares
            if result.non_linear_relationships:
                formatted_text.append("## Relacionamentos Nao-Lineares")
                for relationship in result.non_linear_relationships:
                    formatted_text.append(f"- {relationship}")
                formatted_text.append("")

            # Analise de multicolinearidade
            if result.multicollinearity_analysis:
                formatted_text.append("## Analise de Multicolinearidade")
                multicollinearity = result.multicollinearity_analysis

                if multicollinearity.get('high_correlation_pairs'):
                    formatted_text.append("- Pares com alta correlacao (>0.8):")
                    for pair in multicollinearity['high_correlation_pairs']:
                        formatted_text.append(f"  * {pair}")

                if multicollinearity.get('vif_scores'):
                    formatted_text.append("- Fatores de Inflacao da Variancia (VIF):")
                    for var, vif in multicollinearity['vif_scores'].items():
                        if vif > 5:  # VIF alto
                            formatted_text.append(f"  * {var}: {vif:.2f} (ALTO)")
                        else:
                            formatted_text.append(f"  * {var}: {vif:.2f}")
                formatted_text.append("")

            # Variaveis mais influentes
            if result.most_influential_variables:
                formatted_text.append("## Variaveis Mais Influentes")
                for var_info in result.most_influential_variables:
                    formatted_text.append(f"- {var_info}")
                formatted_text.append("")

            return "\n".join(formatted_text)

        except Exception as e:
            log_error_with_context(e, {"result_type": type(result).__name__}, "relationship_formatting_error")
            raise FormatterError(f"Erro ao formatar analise de relacionamentos: {str(e)}")

    def format_conclusion(self, conclusion: AnalysisConclusion) -> str:
        """
        Formata conclusao da analise em texto legivel.

        Args:
            conclusion: Conclusao da analise

        Returns:
            Texto formatado
        """
        try:
            formatted_text = []

            formatted_text.append("# CONCLUSOES DA ANALISE EDA")
            formatted_text.append("=" * 50)
            formatted_text.append("")

            # Insights principais
            if conclusion.key_insights:
                formatted_text.append("## Insights Principais")
                for i, insight in enumerate(conclusion.key_insights, 1):
                    formatted_text.append(f"{i}. {insight}")
                formatted_text.append("")

            # Descobertas por categoria
            if conclusion.findings_by_category:
                formatted_text.append("## Descobertas por Categoria")
                for category, findings in conclusion.findings_by_category.items():
                    formatted_text.append(f"### {category.replace('_', ' ').title()}")
                    for finding in findings:
                        formatted_text.append(f"- {finding}")
                    formatted_text.append("")

            # Recomendacoes
            if conclusion.recommendations:
                formatted_text.append("## Recomendacoes")
                for i, rec in enumerate(conclusion.recommendations, 1):
                    formatted_text.append(f"{i}. {rec}")
                formatted_text.append("")

            # Limitacoes
            if conclusion.limitations:
                formatted_text.append("## Limitacoes da Analise")
                for limitation in conclusion.limitations:
                    formatted_text.append(f"- {limitation}")
                formatted_text.append("")

            # Proximos passos
            if conclusion.next_steps:
                formatted_text.append("## Proximos Passos Sugeridos")
                for i, step in enumerate(conclusion.next_steps, 1):
                    formatted_text.append(f"{i}. {step}")
                formatted_text.append("")

            # Metadados da analise
            if conclusion.analysis_metadata:
                metadata = conclusion.analysis_metadata
                formatted_text.append("## Informacoes da Analise")
                formatted_text.append(f"- Data/Hora: {metadata.get('timestamp', 'N/A')}")
                formatted_text.append(f"- Agentes utilizados: {', '.join(metadata.get('agents_used', []))}")
                formatted_text.append(f"- Tempo de execucao: {metadata.get('execution_time', 'N/A')}")

            return "\n".join(formatted_text)

        except Exception as e:
            log_error_with_context(e, {"conclusion_type": type(conclusion).__name__}, "conclusion_formatting_error")
            raise FormatterError(f"Erro ao formatar conclusao: {str(e)}")

    def format_statistics_table(self, stats_dict: Dict[str, Any], title: str = "Estatisticas") -> str:
        """
        Formata dicionario de estatisticas em tabela legivel.

        Args:
            stats_dict: Dicionario com estatisticas
            title: Titulo da tabela

        Returns:
            Tabela formatada
        """
        try:
            formatted_text = [f"## {title}", ""]

            # Encontrar largura maxima para alinhamento
            max_key_length = max(len(str(key)) for key in stats_dict.keys()) if stats_dict else 0
            max_value_length = max(len(str(self._format_number(value))) for value in stats_dict.values()) if stats_dict else 0

            # Criar linha separadora
            separator = "-" * (max_key_length + max_value_length + 5)
            formatted_text.append(separator)

            # Adicionar linhas da tabela
            for key, value in stats_dict.items():
                formatted_key = str(key).ljust(max_key_length)
                formatted_value = str(self._format_number(value)).rjust(max_value_length)
                formatted_text.append(f"{formatted_key} | {formatted_value}")

            formatted_text.append(separator)
            formatted_text.append("")

            return "\n".join(formatted_text)

        except Exception as e:
            log_error_with_context(e, {"stats_count": len(stats_dict) if stats_dict else 0}, "statistics_table_formatting_error")
            return f"Erro ao formatar tabela de estatisticas: {str(e)}"

    def format_data_summary(self, df: pd.DataFrame) -> str:
        """
        Cria resumo rapido dos dados em formato legivel.

        Args:
            df: DataFrame para resumir

        Returns:
            Resumo formatado
        """
        try:
            formatted_text = []

            formatted_text.append("# RESUMO RAPIDO DOS DADOS")
            formatted_text.append("=" * 40)
            formatted_text.append("")

            # Informacoes basicas
            formatted_text.append(f"Dimensoes: {df.shape[0]} linhas × {df.shape[1]} colunas")
            formatted_text.append(f"Memoria: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
            formatted_text.append("")

            # Tipos de colunas
            type_counts = df.dtypes.value_counts()
            formatted_text.append("Tipos de dados:")
            for dtype, count in type_counts.items():
                formatted_text.append(f"- {dtype}: {count} colunas")
            formatted_text.append("")

            # Dados ausentes
            missing_data = df.isnull().sum()
            missing_cols = missing_data[missing_data > 0]
            if len(missing_cols) > 0:
                formatted_text.append("Dados ausentes:")
                for col, count in missing_cols.items():
                    pct = (count / len(df)) * 100
                    formatted_text.append(f"- {col}: {count} ({pct:.1f}%)")
            else:
                formatted_text.append("Nenhum dado ausente encontrado")
            formatted_text.append("")

            # Colunas numericas - estatisticas rapidas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                formatted_text.append("Variaveis numericas (primeiras 5):")
                for col in numeric_cols[:5]:
                    serie = df[col]
                    formatted_text.append(f"- {col}: media={serie.mean():.2f}, std={serie.std():.2f}")
                if len(numeric_cols) > 5:
                    formatted_text.append(f"... e mais {len(numeric_cols) - 5} variaveis numericas")
            formatted_text.append("")

            return "\n".join(formatted_text)

        except Exception as e:
            log_error_with_context(e, {"dataframe_shape": df.shape}, "data_summary_formatting_error")
            return f"Erro ao criar resumo dos dados: {str(e)}"

    def _format_number(self, value: Any) -> str:
        """Formata numero para exibicao legivel."""
        if value is None or pd.isna(value):
            return "N/A"

        if isinstance(value, (int, float)):
            if abs(value) >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.1f}K"
            elif isinstance(value, float):
                if abs(value) < 0.001 and value != 0:
                    return f"{value:.2e}"
                else:
                    return f"{value:.3f}".rstrip('0').rstrip('.')
            else:
                return f"{value:,}"

        return str(value)

    def _correlation_strength(self, abs_corr: float) -> str:
        """Determina a forca da correlacao."""
        if abs_corr >= 0.8:
            return "muito forte"
        elif abs_corr >= 0.6:
            return "forte"
        elif abs_corr >= 0.4:
            return "moderada"
        elif abs_corr >= 0.2:
            return "fraca"
        else:
            return "muito fraca"


# Instancia singleton
_response_formatter: Optional[EDAResponseFormatter] = None


def get_response_formatter() -> EDAResponseFormatter:
    """Obtem instancia singleton do formatador de respostas."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = EDAResponseFormatter()
    return _response_formatter


# Funcoes de conveniencia
def format_descriptive_analysis(result: DescriptiveAnalysisResult) -> str:
    """Funcao de conveniencia para formatar analise descritiva."""
    return get_response_formatter().format_descriptive_analysis(result)


def format_data_summary(df: pd.DataFrame) -> str:
    """Funcao de conveniencia para resumo de dados."""
    return get_response_formatter().format_data_summary(df)


def format_statistics_table(stats_dict: Dict[str, Any], title: str = "Estatisticas") -> str:
    """Funcao de conveniencia para tabela de estatisticas."""
    return get_response_formatter().format_statistics_table(stats_dict, title)