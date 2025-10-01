"""
Renderizador especializado de visualizações para Streamlit.

Fornece funcionalidades avançadas para renderizar diferentes tipos
de visualizações e gráficos gerados pelos agentes EDA.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
import base64
import io


class VisualizationRenderer:
    """
    Renderizador especializado para visualizações do EDA Agent.

    Suporta múltiplos tipos de visualização e fornece
    funcionalidades avançadas de renderização e formatação.
    """

    def __init__(self):
        self.supported_types = [
            "plotly", "matplotlib", "seaborn", "dataframe",
            "metric", "text", "json", "table", "chart"
        ]

    def render_visualization(self, visualization: Dict[str, Any]) -> bool:
        """
        Renderiza uma visualização baseada em sua especificação.

        Args:
            visualization: Dicionário com especificação da visualização

        Returns:
            True se renderizado com sucesso, False caso contrário
        """
        try:
            viz_type = visualization.get("type", "").lower()
            viz_data = visualization.get("data")
            viz_title = visualization.get("title", "")
            viz_config = visualization.get("config", {})

            if viz_title:
                st.markdown(f"**{viz_title}**")

            if viz_type == "plotly":
                return self._render_plotly(viz_data, viz_config)
            elif viz_type == "matplotlib":
                return self._render_matplotlib(viz_data, viz_config)
            elif viz_type == "seaborn":
                return self._render_seaborn(viz_data, viz_config)
            elif viz_type == "dataframe":
                return self._render_dataframe(viz_data, viz_config)
            elif viz_type == "metric":
                return self._render_metrics(viz_data, viz_config)
            elif viz_type == "text":
                return self._render_text(viz_data, viz_config)
            elif viz_type == "json":
                return self._render_json(viz_data, viz_config)
            elif viz_type == "table":
                return self._render_table(viz_data, viz_config)
            elif viz_type == "chart":
                return self._render_chart(viz_data, viz_config)
            else:
                st.warning(f"Tipo de visualização não suportado: {viz_type}")
                return False

        except Exception as e:
            st.error(f"Erro ao renderizar visualização: {str(e)}")
            return False

    def _render_plotly(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza gráfico Plotly."""
        try:
            if isinstance(data, (go.Figure, px.scatter, px.line, px.bar)):
                st.plotly_chart(data, width="stretch", **config)
            elif isinstance(data, dict):
                # Criar figura a partir de dados estruturados
                fig = self._create_plotly_from_dict(data)
                if fig:
                    st.plotly_chart(fig, width="stretch", **config)
                else:
                    return False
            else:
                st.error("Dados Plotly inválidos")
                return False
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar Plotly: {str(e)}")
            return False

    def _render_matplotlib(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza gráfico Matplotlib."""
        try:
            if isinstance(data, plt.Figure):
                st.pyplot(data, **config)
            elif isinstance(data, dict):
                # Criar figura a partir de dados
                fig = self._create_matplotlib_from_dict(data)
                if fig:
                    st.pyplot(fig, **config)
                else:
                    return False
            else:
                st.error("Dados Matplotlib inválidos")
                return False
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar Matplotlib: {str(e)}")
            return False

    def _render_seaborn(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza gráfico Seaborn."""
        try:
            # Seaborn usa matplotlib como backend
            if isinstance(data, dict):
                fig = self._create_seaborn_from_dict(data)
                if fig:
                    st.pyplot(fig, **config)
                    return True
            st.error("Dados Seaborn inválidos")
            return False
        except Exception as e:
            st.error(f"Erro ao renderizar Seaborn: {str(e)}")
            return False

    def _render_dataframe(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza DataFrame."""
        try:
            if isinstance(data, pd.DataFrame):
                display_config = {
                    "use_container_width": True,
                    **config
                }
                st.dataframe(data, **display_config)
            elif isinstance(data, dict):
                # Converter dict para DataFrame
                df = pd.DataFrame(data)
                st.dataframe(df, width="stretch", **config)
            else:
                st.error("Dados de DataFrame inválidos")
                return False
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar DataFrame: {str(e)}")
            return False

    def _render_metrics(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza métricas."""
        try:
            if isinstance(data, dict):
                # Organizar métricas em colunas
                metrics = list(data.items())
                cols = st.columns(len(metrics))

                for i, (key, value) in enumerate(metrics):
                    with cols[i]:
                        delta = config.get("deltas", {}).get(key)
                        help_text = config.get("help", {}).get(key)
                        st.metric(
                            label=key,
                            value=value,
                            delta=delta,
                            help=help_text
                        )
            else:
                st.error("Dados de métricas inválidos")
                return False
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar métricas: {str(e)}")
            return False

    def _render_text(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza texto formatado."""
        try:
            text_format = config.get("format", "markdown")

            if text_format == "markdown":
                st.markdown(str(data))
            elif text_format == "code":
                language = config.get("language", "python")
                st.code(str(data), language=language)
            elif text_format == "latex":
                st.latex(str(data))
            else:
                st.text(str(data))
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar texto: {str(e)}")
            return False

    def _render_json(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza dados JSON."""
        try:
            expanded = config.get("expanded", False)
            if expanded:
                st.json(data)
            else:
                with st.expander("Ver JSON", expanded=False):
                    st.json(data)
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar JSON: {str(e)}")
            return False

    def _render_table(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza tabela formatada."""
        try:
            if isinstance(data, (list, dict)):
                st.table(data)
            elif isinstance(data, pd.DataFrame):
                st.table(data)
            else:
                st.error("Dados de tabela inválidos")
                return False
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar tabela: {str(e)}")
            return False

    def _render_chart(self, data: Any, config: Dict[str, Any]) -> bool:
        """Renderiza gráfico usando chart nativo do Streamlit."""
        try:
            chart_type = config.get("chart_type", "line")

            if isinstance(data, pd.DataFrame):
                if chart_type == "line":
                    st.line_chart(data, **config)
                elif chart_type == "area":
                    st.area_chart(data, **config)
                elif chart_type == "bar":
                    st.bar_chart(data, **config)
                else:
                    st.line_chart(data, **config)  # Default
            else:
                st.error("Dados de gráfico inválidos")
                return False
            return True
        except Exception as e:
            st.error(f"Erro ao renderizar gráfico: {str(e)}")
            return False

    def _create_plotly_from_dict(self, data: Dict[str, Any]) -> Optional[go.Figure]:
        """Cria figura Plotly a partir de dicionário."""
        try:
            if "x" in data and "y" in data:
                chart_type = data.get("type", "scatter")

                if chart_type == "scatter":
                    fig = px.scatter(x=data["x"], y=data["y"], title=data.get("title", ""))
                elif chart_type == "line":
                    fig = px.line(x=data["x"], y=data["y"], title=data.get("title", ""))
                elif chart_type == "bar":
                    fig = px.bar(x=data["x"], y=data["y"], title=data.get("title", ""))
                else:
                    fig = px.scatter(x=data["x"], y=data["y"], title=data.get("title", ""))

                return fig

            elif "labels" in data and "values" in data:
                fig = px.pie(values=data["values"], names=data["labels"], title=data.get("title", ""))
                return fig

            return None
        except Exception:
            return None

    def _create_matplotlib_from_dict(self, data: Dict[str, Any]) -> Optional[plt.Figure]:
        """Cria figura Matplotlib a partir de dicionário."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if "x" in data and "y" in data:
                chart_type = data.get("type", "plot")

                if chart_type == "plot":
                    ax.plot(data["x"], data["y"])
                elif chart_type == "scatter":
                    ax.scatter(data["x"], data["y"])
                elif chart_type == "bar":
                    ax.bar(data["x"], data["y"])

                ax.set_title(data.get("title", ""))
                ax.set_xlabel(data.get("xlabel", "X"))
                ax.set_ylabel(data.get("ylabel", "Y"))

            return fig
        except Exception:
            return None

    def _create_seaborn_from_dict(self, data: Dict[str, Any]) -> Optional[plt.Figure]:
        """Cria figura Seaborn a partir de dicionário."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if "dataframe" in data:
                df = data["dataframe"]
                plot_type = data.get("type", "scatter")

                if plot_type == "scatter":
                    sns.scatterplot(data=df, x=data["x"], y=data["y"], ax=ax)
                elif plot_type == "line":
                    sns.lineplot(data=df, x=data["x"], y=data["y"], ax=ax)
                elif plot_type == "hist":
                    sns.histplot(data=df, x=data["x"], ax=ax)
                elif plot_type == "box":
                    sns.boxplot(data=df, x=data["x"], y=data["y"], ax=ax)

                ax.set_title(data.get("title", ""))

            return fig
        except Exception:
            return None

    def render_visualization_gallery(self, visualizations: List[Dict[str, Any]]):
        """
        Renderiza uma galeria de visualizações.

        Args:
            visualizations: Lista de especificações de visualização
        """
        if not visualizations:
            st.info("Nenhuma visualização disponível")
            return

        st.subheader(f"📊 Visualizações ({len(visualizations)})")

        # Organizar em grid se há múltiplas visualizações
        if len(visualizations) == 1:
            self.render_visualization(visualizations[0])
        elif len(visualizations) <= 4:
            # Grid 2x2
            col1, col2 = st.columns(2)
            for i, viz in enumerate(visualizations):
                with col1 if i % 2 == 0 else col2:
                    self.render_visualization(viz)
        else:
            # Tabs para muitas visualizações
            tab_names = [f"Viz {i+1}" for i in range(len(visualizations))]
            tabs = st.tabs(tab_names)

            for tab, viz in zip(tabs, visualizations):
                with tab:
                    self.render_visualization(viz)

    def export_visualization(self, visualization: Dict[str, Any], format: str = "png") -> Optional[str]:
        """
        Exporta visualização para formato especificado.

        Args:
            visualization: Especificação da visualização
            format: Formato de export (png, pdf, svg, html)

        Returns:
            Base64 string da visualização exportada ou None
        """
        try:
            viz_type = visualization.get("type", "").lower()
            viz_data = visualization.get("data")

            if viz_type == "plotly" and isinstance(viz_data, go.Figure):
                if format == "html":
                    return viz_data.to_html()
                else:
                    img_bytes = viz_data.to_image(format=format)
                    return base64.b64encode(img_bytes).decode()

            elif viz_type == "matplotlib" and isinstance(viz_data, plt.Figure):
                buf = io.BytesIO()
                viz_data.savefig(buf, format=format)
                buf.seek(0)
                return base64.b64encode(buf.read()).decode()

            return None
        except Exception:
            return None