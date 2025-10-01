"""
Enumerações para tipos de análise EDA e estados do sistema.
Define constantes para classificação de consultas e controle de fluxo.
"""

from enum import Enum


class EDAAnalysisType(str, Enum):
    """Tipos de análise EDA suportados pelo sistema."""

    DESCRIPTIVE = "descriptive"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    RELATIONSHIP = "relationship"
    CONCLUSION = "conclusion"


class QueryIntentType(str, Enum):
    """Tipos de intenção nas consultas dos usuários."""

    DATA_OVERVIEW = "data_overview"
    STATISTICAL_SUMMARY = "statistical_summary"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    OUTLIER_DETECTION = "outlier_detection"
    PATTERN_DISCOVERY = "pattern_discovery"
    VISUALIZATION_REQUEST = "visualization_request"
    CONCLUSION_REQUEST = "conclusion_request"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON_ANALYSIS = "comparison_analysis"
    SPECIFIC_VALUE_QUERY = "specific_value_query"


class ProcessingStatus(str, Enum):
    """Status de processamento de análises."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class VisualizationType(str, Enum):
    """Tipos de visualizações disponíveis."""

    HISTOGRAM = "histogram"
    BOXPLOT = "boxplot"
    SCATTER = "scatter"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    CORRELATION_HEATMAP = "correlation_heatmap"
    LINE_PLOT = "line_plot"
    BAR_PLOT = "bar_plot"
    BAR_CHART = "bar_chart"
    DISTRIBUTION_PLOT = "distribution_plot"
    TIME_SERIES = "time_series"