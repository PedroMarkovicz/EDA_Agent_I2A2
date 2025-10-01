"""
Ferramentas de visualização para geração automática de gráficos.

Estas ferramentas encapsulam o EDAGraphGenerator e retornam visualizações
em formato base64 que podem ser renderizadas no Streamlit.
"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime

from .base import BaseTool, ToolResult, ToolParameter
from ..utils.graph_generator import get_graph_generator


class PlotDistributionTool(BaseTool):
    """
    Gera histograma/distribuição de uma variável numérica.

    Retorna visualização base64 da distribuição completa da variável,
    incluindo histograma e estatísticas descritivas.
    """

    def __init__(self):
        super().__init__()
        self.name = "plot_distribution"
        self.category = "visualization"
        self.description = (
            "Gera histograma e visualização da distribuição de uma variável numérica. "
            "Use esta ferramenta quando o usuário solicitar: "
            "'distribuição', 'histograma', 'gráfico de X', 'mostre a distribuição', "
            "'como está distribuído', 'plotar X'. "
            "Retorna gráfico em formato base64 pronto para renderização."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para plotar distribuição",
                required=True
            ),
            ToolParameter(
                name="bins",
                type="integer",
                description="Número de bins do histograma (padrão: 30)",
                required=False
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')
        bins = kwargs.get('bins', 30)

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica. Use gráfico de barras para variáveis categóricas.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            graph_generator = get_graph_generator()

            # Criar histograma usando EDAGraphGenerator
            hist_result = graph_generator.create_histogram(
                data=col_data,
                title=f"Distribuição de {column}",
                bins=bins,
                show_stats=True
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Estruturar resultado no formato esperado
            result_data = {
                "plot_type": "histogram",
                "column": column,
                "title": hist_result.get("title"),
                "image_base64": hist_result.get("image_base64"),
                "stats": hist_result.get("stats", {}),
                "bins": bins
            }

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao gerar histograma: {str(e)}")


class PlotBoxplotTool(BaseTool):
    """
    Gera boxplot de uma variável numérica para análise de outliers.

    Retorna visualização base64 do boxplot com informações sobre outliers.
    """

    def __init__(self):
        super().__init__()
        self.name = "plot_boxplot"
        self.category = "visualization"
        self.description = (
            "Gera boxplot de uma variável numérica para detectar outliers e analisar dispersão. "
            "Use esta ferramenta quando o usuário solicitar: "
            "'boxplot', 'diagrama de caixa', 'outliers', 'valores atípicos', "
            "'dispersão de X', 'quartis'. "
            "Retorna gráfico em formato base64 pronto para renderização."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna para plotar boxplot",
                required=True
            ),
            ToolParameter(
                name="orientation",
                type="string",
                description="Orientação do boxplot: 'v' (vertical) ou 'h' (horizontal). Padrão: 'v'",
                required=False
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')
        orientation = kwargs.get('orientation', 'v')

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return self._create_error_result(f"Coluna '{column}' não é numérica.")

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            graph_generator = get_graph_generator()

            # Criar boxplot usando EDAGraphGenerator
            box_result = graph_generator.create_boxplot(
                data=col_data,
                title=f"Boxplot de {column}",
                orient=orientation
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Estruturar resultado no formato esperado
            result_data = {
                "plot_type": "boxplot",
                "column": column,
                "title": box_result.get("title"),
                "image_base64": box_result.get("image_base64"),
                "outliers_info": box_result.get("outliers_info", {}),
                "orientation": orientation
            }

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao gerar boxplot: {str(e)}")


class PlotCorrelationHeatmapTool(BaseTool):
    """
    Gera heatmap de correlação entre variáveis numéricas do dataset.

    Retorna visualização base64 da matriz de correlação completa.
    """

    def __init__(self):
        super().__init__()
        self.name = "plot_correlation_heatmap"
        self.category = "visualization"
        self.description = (
            "Gera heatmap (mapa de calor) da matriz de correlação entre todas as variáveis numéricas. "
            "Use esta ferramenta quando o usuário solicitar: "
            "'correlação', 'matriz de correlação', 'heatmap', 'mapa de calor', "
            "'relação entre variáveis', 'quais variáveis estão correlacionadas'. "
            "Retorna gráfico em formato base64 pronto para renderização."
        )
        self.parameters = [
            ToolParameter(
                name="method",
                type="string",
                description="Método de correlação: 'pearson', 'spearman', ou 'kendall'. Padrão: 'pearson'",
                required=False
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        method = kwargs.get('method', 'pearson')

        # Validar método
        if method not in ['pearson', 'spearman', 'kendall']:
            return self._create_error_result(f"Método '{method}' inválido. Use 'pearson', 'spearman', ou 'kendall'.")

        # Verificar se há variáveis numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return self._create_error_result("Dataset precisa de pelo menos 2 variáveis numéricas para calcular correlação.")

        try:
            graph_generator = get_graph_generator()

            # Criar heatmap de correlação usando EDAGraphGenerator
            heatmap_result = graph_generator.create_correlation_heatmap(
                data=df,
                title=f"Matriz de Correlação ({method.capitalize()})",
                method=method
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Estruturar resultado no formato esperado
            result_data = {
                "plot_type": "correlation_heatmap",
                "title": heatmap_result.get("title"),
                "image_base64": heatmap_result.get("image_base64"),
                "correlation_matrix": heatmap_result.get("correlation_matrix", {}),
                "strongest_correlations": heatmap_result.get("strongest_correlations", []),
                "method": method,
                "num_variables": len(numeric_cols)
            }

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao gerar heatmap de correlação: {str(e)}")


class PlotBarChartTool(BaseTool):
    """
    Gera gráfico de barras para variáveis categóricas.

    Retorna visualização base64 da distribuição de frequências.
    """

    def __init__(self):
        super().__init__()
        self.name = "plot_bar_chart"
        self.category = "visualization"
        self.description = (
            "Gera gráfico de barras para variáveis categóricas mostrando frequências. "
            "Use esta ferramenta quando o usuário solicitar: "
            "'gráfico de barras', 'frequência de X', 'contagem por categoria', "
            "'distribuição categórica'. "
            "Retorna gráfico em formato base64 pronto para renderização."
        )
        self.parameters = [
            ToolParameter(
                name="column",
                type="string",
                description="Nome da coluna categórica para plotar",
                required=True
            ),
            ToolParameter(
                name="max_categories",
                type="integer",
                description="Número máximo de categorias a exibir (padrão: 20)",
                required=False
            )
        ]

    def execute(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        start_time = datetime.now()
        column = kwargs.get('column')
        max_categories = kwargs.get('max_categories', 20)

        if column not in df.columns:
            return self._create_error_result(f"Coluna '{column}' não encontrada.")

        col_data = df[column]

        if col_data.empty:
            return self._create_error_result(f"Coluna '{column}' não contém valores válidos.")

        try:
            graph_generator = get_graph_generator()

            # Criar gráfico de barras usando EDAGraphGenerator
            bar_result = graph_generator.create_bar_chart(
                data=col_data,
                title=f"Distribuição de {column}",
                max_categories=max_categories,
                sort_values=True
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Estruturar resultado no formato esperado
            result_data = {
                "plot_type": "bar_chart",
                "column": column,
                "title": bar_result.get("title"),
                "image_base64": bar_result.get("image_base64"),
                "value_counts": bar_result.get("value_counts", {}),
                "total_categories": bar_result.get("total_categories", 0),
                "displayed_categories": bar_result.get("displayed_categories", 0)
            }

            return self._create_success_result(
                data=result_data,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_error_result(f"Erro ao gerar gráfico de barras: {str(e)}")
