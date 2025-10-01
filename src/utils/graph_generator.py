"""
Gerador de graficos e visualizacoes EDA padronizadas.
Suporta histogramas, boxplots, scatter plots, correlation heatmaps
e outras visualizacoes essenciais para analise exploratoria de dados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import warnings

from ..core.config import get_config
from ..core.logger import get_logger, log_data_operation, log_error_with_context
from ..models.enums import VisualizationType

warnings.filterwarnings('ignore')


class GraphGeneratorError(Exception):
    """Erro especifico do gerador de graficos."""
    pass


class EDAGraphGenerator:
    """Gerador de visualizacoes para analise exploratoria de dados."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("graph_generator")
        self.setup_matplotlib_style()
        self.setup_seaborn_style()

    def setup_matplotlib_style(self):
        """Configura estilo padrao para matplotlib."""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight'
        })

    def setup_seaborn_style(self):
        """Configura estilo padrao para seaborn."""
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def create_histogram(self,
                        data: pd.Series,
                        title: Optional[str] = None,
                        bins: int = 30,
                        show_stats: bool = True) -> Dict[str, Any]:
        """
        Cria histograma para variavel numerica.

        Args:
            data: Serie com dados numericos
            title: Titulo do grafico
            bins: Numero de bins
            show_stats: Se deve mostrar estatisticas

        Returns:
            Dicionario com informacoes do grafico
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Remover valores nulos
            clean_data = data.dropna()

            if len(clean_data) == 0:
                raise GraphGeneratorError("Nenhum dado valido para histograma")

            # Criar histograma
            n, bins_used, patches = ax.hist(clean_data, bins=bins, alpha=0.7, edgecolor='black')

            # Configurar titulo
            if title is None:
                title = f"Histograma de {data.name or 'Variavel'}"
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(data.name or 'Valores')
            ax.set_ylabel('Frequencia')

            # Adicionar estatisticas se solicitado
            if show_stats:
                stats_text = (
                    f"Media: {clean_data.mean():.2f}\n"
                    f"Mediana: {clean_data.median():.2f}\n"
                    f"Desvio: {clean_data.std():.2f}\n"
                    f"N: {len(clean_data)}"
                )
                ax.text(0.75, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            result = {
                "type": VisualizationType.HISTOGRAM,
                "title": title,
                "image_base64": img_str,
                "stats": {
                    "mean": float(clean_data.mean()),
                    "median": float(clean_data.median()),
                    "std": float(clean_data.std()),
                    "count": len(clean_data),
                    "missing": len(data) - len(clean_data)
                }
            }

            log_data_operation(
                "histogram_created",
                {"column": data.name, "bins": bins, "data_points": len(clean_data)},
                "graph_generator"
            )

            return result

        except Exception as e:
            log_error_with_context(
                e,
                {"column": data.name, "data_length": len(data)},
                "graph_generator"
            )
            raise GraphGeneratorError(f"Erro ao criar histograma: {str(e)}")

    def create_boxplot(self,
                      data: Union[pd.Series, pd.DataFrame],
                      title: Optional[str] = None,
                      orient: str = 'v') -> Dict[str, Any]:
        """
        Cria boxplot para detectar outliers.

        Args:
            data: Dados para o boxplot
            title: Titulo do grafico
            orient: Orientacao ('v' vertical, 'h' horizontal)

        Returns:
            Dicionario com informacoes do grafico
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(data, pd.Series):
                clean_data = data.dropna()
                if orient == 'v':
                    box_plot = ax.boxplot([clean_data], labels=[data.name or 'Dados'])
                else:
                    box_plot = ax.boxplot([clean_data], vert=False, labels=[data.name or 'Dados'])
            else:
                # DataFrame com multiplas colunas numericas
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                clean_data = data[numeric_cols].dropna()

                if orient == 'v':
                    box_plot = ax.boxplot([clean_data[col] for col in numeric_cols], labels=numeric_cols)
                else:
                    box_plot = ax.boxplot([clean_data[col] for col in numeric_cols], vert=False, labels=numeric_cols)

            if title is None:
                title = f"Boxplot de {data.name if isinstance(data, pd.Series) else 'Variaveis Numericas'}"
            ax.set_title(title, fontweight='bold')

            if orient == 'v':
                ax.set_ylabel('Valores')
                plt.xticks(rotation=45)
            else:
                ax.set_xlabel('Valores')

            plt.tight_layout()

            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            # Calcular outliers se for Serie
            outliers_info = {}
            if isinstance(data, pd.Series):
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]

                outliers_info = {
                    "count": len(outliers),
                    "percentage": (len(outliers) / len(data)) * 100,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }

            result = {
                "type": VisualizationType.BOXPLOT,
                "title": title,
                "image_base64": img_str,
                "outliers_info": outliers_info
            }

            log_data_operation(
                "boxplot_created",
                {"data_type": "series" if isinstance(data, pd.Series) else "dataframe"},
                "graph_generator"
            )

            return result

        except Exception as e:
            log_error_with_context(
                e,
                {"data_shape": data.shape},
                "graph_generator"
            )
            raise GraphGeneratorError(f"Erro ao criar boxplot: {str(e)}")

    def create_scatter_plot(self,
                           x_data: pd.Series,
                           y_data: pd.Series,
                           title: Optional[str] = None,
                           add_regression: bool = True) -> Dict[str, Any]:
        """
        Cria scatter plot para analise de relacionamentos.

        Args:
            x_data: Dados do eixo X
            y_data: Dados do eixo Y
            title: Titulo do grafico
            add_regression: Se deve adicionar linha de regressao

        Returns:
            Dicionario com informacoes do grafico
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Remover pares com valores nulos
            combined_data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()

            if len(combined_data) == 0:
                raise GraphGeneratorError("Nenhum par de dados valido para scatter plot")

            x_clean = combined_data['x']
            y_clean = combined_data['y']

            # Criar scatter plot
            ax.scatter(x_clean, y_clean, alpha=0.6, s=50)

            # Adicionar linha de regressao se solicitado
            correlation = 0
            if add_regression and len(x_clean) > 1:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                ax.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2)

                # Calcular correlacao
                correlation = x_clean.corr(y_clean)

            if title is None:
                title = f"Scatter Plot: {x_data.name or 'X'} vs {y_data.name or 'Y'}"
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(x_data.name or 'X')
            ax.set_ylabel(y_data.name or 'Y')

            # Adicionar informacao de correlacao
            if add_regression:
                ax.text(0.05, 0.95, f'Correlacao: {correlation:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()

            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            result = {
                "type": VisualizationType.SCATTER,
                "title": title,
                "image_base64": img_str,
                "correlation": float(correlation),
                "data_points": len(combined_data)
            }

            log_data_operation(
                "scatter_plot_created",
                {"x_column": x_data.name, "y_column": y_data.name, "correlation": correlation},
                "graph_generator"
            )

            return result

        except Exception as e:
            log_error_with_context(
                e,
                {"x_column": x_data.name, "y_column": y_data.name},
                "graph_generator"
            )
            raise GraphGeneratorError(f"Erro ao criar scatter plot: {str(e)}")

    def create_correlation_heatmap(self,
                                  data: pd.DataFrame,
                                  title: Optional[str] = None,
                                  method: str = 'pearson') -> Dict[str, Any]:
        """
        Cria heatmap de correlacao entre variaveis.

        Args:
            data: DataFrame com variaveis numericas
            title: Titulo do grafico
            method: Metodo de correlacao ('pearson', 'spearman', 'kendall')

        Returns:
            Dicionario com informacoes do grafico
        """
        try:
            # Selecionar apenas colunas numericas
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                raise GraphGeneratorError("Nenhuma variavel numerica encontrada para correlacao")

            # Calcular matriz de correlacao
            corr_matrix = numeric_data.corr(method=method)

            # Criar heatmap
            fig, ax = plt.subplots(figsize=(12, 10))

            # Usar seaborn para heatmap mais bonito
            sns.heatmap(corr_matrix,
                       annot=True,
                       cmap='coolwarm',
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={"shrink": .8},
                       ax=ax)

            if title is None:
                title = f"Matriz de Correlacao ({method.capitalize()})"
            ax.set_title(title, fontweight='bold', pad=20)

            plt.tight_layout()

            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            # Encontrar correlacoes mais fortes
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.1:  # Apenas correlacoes significativas
                        corr_pairs.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": float(corr_value)
                        })

            # Ordenar por valor absoluto de correlacao
            corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            result = {
                "type": VisualizationType.HEATMAP,
                "title": title,
                "image_base64": img_str,
                "correlation_matrix": corr_matrix.to_dict(),
                "strongest_correlations": corr_pairs[:10],  # Top 10
                "method": method
            }

            log_data_operation(
                "correlation_heatmap_created",
                {"variables": len(corr_matrix.columns), "method": method},
                "graph_generator"
            )

            return result

        except Exception as e:
            log_error_with_context(
                e,
                {"data_shape": data.shape, "method": method},
                "graph_generator"
            )
            raise GraphGeneratorError(f"Erro ao criar heatmap de correlacao: {str(e)}")

    def create_bar_chart(self,
                        data: pd.Series,
                        title: Optional[str] = None,
                        max_categories: int = 20,
                        sort_values: bool = True) -> Dict[str, Any]:
        """
        Cria grafico de barras para variaveis categoricas.

        Args:
            data: Serie com dados categoricos
            title: Titulo do grafico
            max_categories: Numero maximo de categorias a exibir
            sort_values: Se deve ordenar por frequencia

        Returns:
            Dicionario com informacoes do grafico
        """
        try:
            # Contar frequencias
            value_counts = data.value_counts()

            # Limitar numero de categorias
            if len(value_counts) > max_categories:
                value_counts = value_counts.head(max_categories)

            if sort_values:
                value_counts = value_counts.sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(12, 8))

            # Criar grafico de barras horizontal para melhor legibilidade
            bars = ax.barh(range(len(value_counts)), value_counts.values)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels(value_counts.index)

            if title is None:
                title = f"Distribuicao de {data.name or 'Categorias'}"
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Frequencia')

            # Adicionar valores nas barras
            for i, (bar, value) in enumerate(zip(bars, value_counts.values)):
                ax.text(bar.get_width() + max(value_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                       str(value), ha='left', va='center')

            plt.tight_layout()

            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            result = {
                "type": VisualizationType.BAR_CHART,
                "title": title,
                "image_base64": img_str,
                "value_counts": value_counts.to_dict(),
                "total_categories": len(data.value_counts()),
                "displayed_categories": len(value_counts)
            }

            log_data_operation(
                "bar_chart_created",
                {"column": data.name, "categories": len(value_counts)},
                "graph_generator"
            )

            return result

        except Exception as e:
            log_error_with_context(
                e,
                {"column": data.name, "unique_values": data.nunique()},
                "graph_generator"
            )
            raise GraphGeneratorError(f"Erro ao criar grafico de barras: {str(e)}")

    def create_time_series_plot(self,
                               data: pd.DataFrame,
                               date_column: str,
                               value_columns: List[str],
                               title: Optional[str] = None) -> Dict[str, Any]:
        """
        Cria grafico de serie temporal.

        Args:
            data: DataFrame com dados
            date_column: Nome da coluna de data
            value_columns: Lista de colunas de valores
            title: Titulo do grafico

        Returns:
            Dicionario com informacoes do grafico
        """
        try:
            # Preparar dados
            plot_data = data[[date_column] + value_columns].copy()

            # Converter coluna de data
            plot_data[date_column] = pd.to_datetime(plot_data[date_column])
            plot_data = plot_data.sort_values(date_column)

            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot each value column
            for col in value_columns:
                ax.plot(plot_data[date_column], plot_data[col], label=col, linewidth=2)

            if title is None:
                title = "Serie Temporal"
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('Valores')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Rotacionar labels de data
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            result = {
                "type": VisualizationType.TIME_SERIES,
                "title": title,
                "image_base64": img_str,
                "date_range": {
                    "start": plot_data[date_column].min().isoformat(),
                    "end": plot_data[date_column].max().isoformat()
                },
                "value_columns": value_columns
            }

            log_data_operation(
                "time_series_plot_created",
                {"date_column": date_column, "value_columns": value_columns},
                "graph_generator"
            )

            return result

        except Exception as e:
            log_error_with_context(
                e,
                {"date_column": date_column, "value_columns": value_columns},
                "graph_generator"
            )
            raise GraphGeneratorError(f"Erro ao criar grafico de serie temporal: {str(e)}")

    def save_graph_to_file(self, graph_data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """
        Salva grafico em arquivo.

        Args:
            graph_data: Dados do grafico com image_base64
            file_path: Caminho para salvar o arquivo

        Returns:
            True se salvou com sucesso
        """
        try:
            file_path = Path(file_path)

            # Decodificar imagem base64
            img_data = base64.b64decode(graph_data["image_base64"])

            # Salvar arquivo
            with open(file_path, 'wb') as f:
                f.write(img_data)

            log_data_operation(
                "graph_saved_to_file",
                {"file_path": str(file_path), "graph_type": graph_data.get("type")},
                "graph_generator"
            )

            return True

        except Exception as e:
            log_error_with_context(
                e,
                {"file_path": str(file_path)},
                "graph_generator"
            )
            return False


# Instancia singleton
_graph_generator: Optional[EDAGraphGenerator] = None


def get_graph_generator() -> EDAGraphGenerator:
    """Obtem instancia singleton do gerador de graficos."""
    global _graph_generator
    if _graph_generator is None:
        _graph_generator = EDAGraphGenerator()
    return _graph_generator


# Funcoes de conveniencia
def create_histogram(data: pd.Series, **kwargs) -> Dict[str, Any]:
    """Funcao de conveniencia para criar histograma."""
    return get_graph_generator().create_histogram(data, **kwargs)


def create_boxplot(data: Union[pd.Series, pd.DataFrame], **kwargs) -> Dict[str, Any]:
    """Funcao de conveniencia para criar boxplot."""
    return get_graph_generator().create_boxplot(data, **kwargs)


def create_scatter_plot(x_data: pd.Series, y_data: pd.Series, **kwargs) -> Dict[str, Any]:
    """Funcao de conveniencia para criar scatter plot."""
    return get_graph_generator().create_scatter_plot(x_data, y_data, **kwargs)


def create_correlation_heatmap(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Funcao de conveniencia para criar heatmap de correlacao."""
    return get_graph_generator().create_correlation_heatmap(data, **kwargs)