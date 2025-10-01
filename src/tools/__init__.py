"""
Tools module - Biblioteca de ferramentas estatísticas para análise EDA.

Este módulo fornece ferramentas estatísticas que podem ser chamadas
dinamicamente pelo LLM orquestrador, eliminando a necessidade de
hardcoding e regex para interpretação de queries.
"""

from .registry import ToolRegistry, get_tool_registry
from .base import BaseTool, ToolResult, ToolParameter
from .basic_stats import BasicStatsTool
from .correlation_analysis import CorrelationTool
from .outlier_detection import OutlierDetectionTool
from .missing_data_analysis import MissingDataTool
from .schema_analysis import SchemaAnalysisTool
from .visualization_tools import (
    PlotDistributionTool,
    PlotBoxplotTool,
    PlotCorrelationHeatmapTool,
    PlotBarChartTool
)
from .minimal_stats import (
    GetMeanTool,
    GetMedianTool,
    GetMaxTool,
    GetMinTool,
    GetSumTool,
    GetStdTool,
    GetCountTool
)

__all__ = [
    'ToolRegistry',
    'get_tool_registry',
    'BaseTool',
    'ToolResult',
    'ToolParameter',
    'BasicStatsTool',
    'CorrelationTool',
    'OutlierDetectionTool',
    'MissingDataTool',
    'SchemaAnalysisTool',
    'PlotDistributionTool',
    'PlotBoxplotTool',
    'PlotCorrelationHeatmapTool',
    'PlotBarChartTool',
    'GetMeanTool',
    'GetMedianTool',
    'GetMaxTool',
    'GetMinTool',
    'GetSumTool',
    'GetStdTool',
    'GetCountTool',
]
