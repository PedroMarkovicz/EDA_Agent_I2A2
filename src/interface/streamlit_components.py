"""
Componentes reutilizáveis para interface Streamlit.

Fornece widgets e componentes especializados para exibição
de dados, métricas e análises no contexto do EDA Agent.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
from datetime import datetime


def render_metric_card(title: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """
    Renderiza um cartão de métrica personalizado.

    Args:
        title: Título da métrica
        value: Valor principal
        delta: Valor de mudança (opcional)
        help_text: Texto de ajuda (opcional)
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )


def render_analysis_summary(analysis_results: Dict[str, Any]):
    """
    Renderiza um resumo dos resultados de análise.

    Args:
        analysis_results: Dicionário com resultados das análises
    """
    if not analysis_results:
        st.info("Nenhum resultado de análise disponível")
        return

    st.subheader("📊 Resumo da Análise")

    # Criar colunas para métricas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        descriptive_count = len(analysis_results.get("descriptive", {}))
        render_metric_card("Análises Descritivas", str(descriptive_count))

    with col2:
        pattern_count = len(analysis_results.get("pattern_analysis", {}))
        render_metric_card("Padrões Detectados", str(pattern_count))

    with col3:
        outlier_count = len(analysis_results.get("outlier_detection", {}))
        render_metric_card("Outliers Encontrados", str(outlier_count))

    with col4:
        correlation_count = len(analysis_results.get("correlation", {}))
        render_metric_card("Correlações", str(correlation_count))

    # Mostrar detalhes se expandido
    with st.expander("Ver detalhes completos", expanded=False):
        for analysis_type, results in analysis_results.items():
            if results:
                st.markdown(f"**{analysis_type.replace('_', ' ').title()}:**")
                if isinstance(results, dict):
                    st.json(results)
                else:
                    st.write(results)


def render_data_preview(df: pd.DataFrame, title: str = "Preview dos Dados"):
    """
    Renderiza um preview inteligente dos dados.

    Args:
        df: DataFrame para exibir
        title: Título da seção
    """
    st.subheader(title)

    # Informações básicas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Linhas", f"{len(df):,}")
    with col2:
        st.metric("Colunas", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memória", f"{memory_mb:.1f} MB")

    # Análise rápida dos tipos de dados
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Colunas Numéricas", len(numeric_cols))
    with col2:
        st.metric("Colunas Categóricas", len(categorical_cols))
    with col3:
        st.metric("Colunas de Data", len(datetime_cols))

    # Preview das primeiras linhas
    with st.expander("Primeiras 10 linhas", expanded=True):
        st.dataframe(df.head(10), width="stretch")

    # Informações das colunas
    with st.expander("Informações das colunas", expanded=False):
        col_info = pd.DataFrame({
            'Tipo': df.dtypes.astype(str),  # Converter dtypes para string para compatibilidade com Arrow
            'Não-Nulos': df.count(),
            'Nulos': df.isnull().sum(),
            '% Nulos': (df.isnull().sum() / len(df) * 100).round(2),
            'Únicos': df.nunique()
        })
        st.dataframe(col_info, width="stretch")

    # Estatísticas descritivas para colunas numéricas
    if len(numeric_cols) > 0:
        with st.expander("Estatísticas descritivas", expanded=False):
            st.dataframe(df[numeric_cols].describe(), width="stretch")


def render_query_suggestions(df: Optional[pd.DataFrame] = None):
    """
    Renderiza sugestões de consultas baseadas nos dados.

    Args:
        df: DataFrame para analisar (opcional)
    """
    st.subheader("💡 Sugestões de Perguntas")

    if df is None:
        suggestions = [
            "Como posso começar a análise exploratória?",
            "Quais tipos de análise estão disponíveis?",
            "Preciso de dados CSV para usar o sistema?"
        ]
    else:
        # Gerar sugestões baseadas nos dados
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        suggestions = []

        if len(numeric_cols) > 0:
            suggestions.extend([
                f"Quais são as estatísticas descritivas de {numeric_cols[0]}?",
                "Há outliers nos dados numéricos?",
                "Qual a distribuição das variáveis numéricas?"
            ])

        if len(numeric_cols) >= 2:
            suggestions.append(f"Existe correlação entre {numeric_cols[0]} e {numeric_cols[1]}?")

        if len(categorical_cols) > 0:
            suggestions.extend([
                f"Quais são os valores mais frequentes em {categorical_cols[0]}?",
                "Como estão distribuídas as categorias?"
            ])

        suggestions.extend([
            "Gere visualizações para melhor entendimento dos dados",
            "Identifique padrões interessantes nos dados",
            "Faça uma análise exploratória completa"
        ])

    # Exibir sugestões como botões clicáveis
    for i, suggestion in enumerate(suggestions[:6]):  # Limitar a 6 sugestões
        if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
            # Copiar sugestão para área de texto (seria implementado via JavaScript)
            st.session_state.suggested_query = suggestion
            st.info(f"Sugestão selecionada: {suggestion}")


def render_progress_indicator(current_step: int, total_steps: int, step_name: str):
    """
    Renderiza um indicador de progresso para análises.

    Args:
        current_step: Passo atual
        total_steps: Total de passos
        step_name: Nome do passo atual
    """
    progress = current_step / total_steps
    st.progress(progress, text=f"Passo {current_step}/{total_steps}: {step_name}")


def render_error_details(error_info: Dict[str, Any]):
    """
    Renderiza detalhes de erro de forma organizada.

    Args:
        error_info: Informações do erro
    """
    st.error("❌ Erro na Análise")

    with st.expander("Detalhes do erro", expanded=False):
        st.markdown(f"**Tipo:** {error_info.get('type', 'Desconhecido')}")
        st.markdown(f"**Mensagem:** {error_info.get('message', 'Sem mensagem')}")

        if error_info.get('timestamp'):
            st.markdown(f"**Horário:** {error_info['timestamp']}")

        if error_info.get('context'):
            st.markdown("**Contexto:**")
            st.json(error_info['context'])

        if error_info.get('suggestions'):
            st.markdown("**Sugestões:**")
            for suggestion in error_info['suggestions']:
                st.markdown(f"• {suggestion}")


def render_loading_state(message: str = "Processando..."):
    """
    Renderiza um estado de carregamento personalizado.

    Args:
        message: Mensagem a exibir
    """
    with st.spinner(message):
        # Placeholder para animação de carregamento
        placeholder = st.empty()
        return placeholder


def render_success_notification(message: str, details: Optional[Dict[str, Any]] = None):
    """
    Renderiza uma notificação de sucesso.

    Args:
        message: Mensagem principal
        details: Detalhes adicionais (opcional)
    """
    st.success(f"✅ {message}")

    if details:
        with st.expander("Ver detalhes", expanded=False):
            for key, value in details.items():
                st.markdown(f"**{key}:** {value}")