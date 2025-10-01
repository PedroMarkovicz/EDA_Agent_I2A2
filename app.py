"""
Sistema EDA Agent - Interface Streamlit

Interface web para interação com o sistema multi-agente de análise exploratória de dados.
Permite upload de arquivos CSV e consultas em linguagem natural para análises EDA.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List
import io
import traceback
from datetime import datetime
import logging

# Importações do sistema EDA
try:
    from src.graph import execute_eda_analysis_sync, get_module_info
    from src.utils.validators import CSVValidator
    from src.utils.formatters import EDAResponseFormatter
    from src.core.logger import get_logger
    from src.core.config import get_config
except ImportError as e:
    st.error(f"Erro ao importar módulos do sistema EDA: {str(e)}")
    st.stop()


def sanitize_dataframe_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitiza DataFrame para compatibilidade com Arrow/Streamlit.

    Args:
        df: DataFrame a ser sanitizado

    Returns:
        DataFrame com tipos compatíveis com Arrow
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return df

    # Criar cópia para não modificar o original
    df_copy = df.copy()

    # Converter colunas com tipos incompatíveis
    for col in df_copy.columns:
        # Verificar se a coluna contém dtypes do numpy (que não são serializáveis)
        if df_copy[col].dtype == object:
            try:
                # Tentar converter para string se houver objetos não-serializáveis
                if any(hasattr(val, 'dtype') for val in df_copy[col].dropna().head()):
                    df_copy[col] = df_copy[col].astype(str)
            except Exception:
                # Se falhar, manter como está
                pass

    return df_copy


class StreamlitEDAInterface:
    """
    Interface principal do sistema EDA Agent para Streamlit.

    Gerencia todo o ciclo de vida da aplicação incluindo:
    - Configuração da página
    - Gerenciamento de sessão
    - Interface de upload de CSV
    - Sistema de chat para consultas EDA
    - Renderização de resultados e visualizações
    """

    def __init__(self):
        """Inicializa a interface Streamlit."""
        self.setup_page_config()
        self.initialize_session_state()
        self.logger = get_logger("streamlit_interface")

    def setup_page_config(self):
        """Configura a página Streamlit."""
        st.set_page_config(
            page_title="EDA Agent - Análise Exploratória de Dados",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': "EDA Agent - Sistema multi-agente para análise exploratória de dados"
            }
        )

    def initialize_session_state(self):
        """Inicializa o estado da sessão Streamlit."""
        # Estado do CSV
        if 'csv_data' not in st.session_state:
            st.session_state.csv_data = None
        if 'csv_metadata' not in st.session_state:
            st.session_state.csv_metadata = None
        if 'csv_filename' not in st.session_state:
            st.session_state.csv_filename = None

        # Histórico de chat
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Estado do sistema
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'last_analysis_result' not in st.session_state:
            st.session_state.last_analysis_result = None

        # Configurações da interface
        if 'show_debug_info' not in st.session_state:
            st.session_state.show_debug_info = False

        # Cache de resultados para performance
        if 'result_cache' not in st.session_state:
            st.session_state.result_cache = {}

        # Configurações de sessão
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = datetime.now().isoformat()

        # ID único da sessão para logging e memória contextual
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())[:8]

        # Criar sessão de memória contextual se não existir
        if 'memory_session_created' not in st.session_state:
            try:
                from src.core.memory_manager import get_memory_manager
                memory_manager = get_memory_manager()
                # Tentar recuperar sessão existente ou criar nova
                session = memory_manager.get_session(st.session_state.session_id)
                if session is None:
                    # Criar nova sessão no MemoryManager
                    memory_session_id = memory_manager.create_session(user_id=None)
                    # Atualizar session_id para usar o do MemoryManager
                    st.session_state.session_id = memory_session_id
                st.session_state.memory_session_created = True
            except Exception as e:
                self.logger.warning(f"Erro ao criar sessão de memória: {e}")
                st.session_state.memory_session_created = False

    def render_header(self):
        """Renderiza o cabeçalho da aplicação."""
        # Layout com logo e título
        col1, col2 = st.columns([1, 4])

        with col1:
            try:
                st.image("logo-agente-aprende.png", width=150)
            except Exception as e:
                # Se o logo não estiver disponível, continuar sem erro
                pass

        with col2:
            st.title("📊 EDA Agent")
            st.markdown("""
            **Sistema Inteligente de Análise Exploratória de Dados**

            Faça upload de um arquivo CSV e converse com nossos agentes especializados para obter
            insights valiosos sobre seus dados através de análises exploratórias avançadas.
            """)

        st.divider()

    def render_sidebar(self):
        """Renderiza a barra lateral com informações e configurações."""
        with st.sidebar:
            st.header("⚙️ Configurações")

            # Informações do sistema
            st.subheader("Sistema")
            try:
                module_info = get_module_info()
                if module_info.get('is_valid', False):
                    st.success("Sistema operacional")
                    st.text(f"Versão: {module_info.get('version', 'N/A')}")
                    st.text(f"Nós do workflow: {module_info.get('total_nodes', 0)}")
                else:
                    st.warning("Sistema com problemas")
                    if 'error' in module_info:
                        st.error(f"Erro: {module_info['error']}")
            except Exception as e:
                st.error(f"Erro ao verificar sistema: {str(e)}")

            st.divider()

            # Configurações de debug
            st.subheader("Debug")
            st.session_state.show_debug_info = st.checkbox(
                "Mostrar informações de debug",
                value=st.session_state.show_debug_info
            )

            # Informações do CSV carregado
            if st.session_state.csv_data is not None:
                st.subheader("📁 Arquivo Carregado")
                st.text(f"Nome: {st.session_state.csv_filename}")
                st.text(f"Linhas: {len(st.session_state.csv_data)}")
                st.text(f"Colunas: {len(st.session_state.csv_data.columns)}")

                if st.button("🗑️ Remover arquivo"):
                    self.clear_csv_data()
                    st.rerun()

            st.divider()

            # Histórico de análises
            if st.session_state.chat_history:
                st.subheader("📈 Histórico")
                st.text(f"Consultas: {len(st.session_state.chat_history)}")

                if st.button("🧹 Limpar histórico"):
                    st.session_state.chat_history = []
                    st.session_state.result_cache = {}
                    st.rerun()

            st.divider()

            # Informações da sessão
            st.subheader("🔗 Sessão")
            start_time = datetime.fromisoformat(st.session_state.session_start_time)
            duration = datetime.now() - start_time
            duration_str = f"{int(duration.total_seconds() // 60)}m {int(duration.total_seconds() % 60)}s"

            st.text(f"ID: {st.session_state.session_id}")
            st.text(f"Duração: {duration_str}")
            st.text(f"Cache: {len(st.session_state.result_cache)} itens")

            # Botão de reset geral
            st.divider()
            if st.button("🔄 Reset Completo", type="secondary"):
                self.reset_application()
                st.rerun()

    def clear_csv_data(self):
        """Remove dados do CSV da sessão."""
        st.session_state.csv_data = None
        st.session_state.csv_metadata = None
        st.session_state.csv_filename = None

    def reset_application(self):
        """Reseta completamente a aplicação."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self.initialize_session_state()

    def run(self):
        """Executa a aplicação principal."""
        try:
            self.render_header()
            self.render_sidebar()

            # Seção superior: Upload e Preview lado a lado
            st.markdown("### 📂 Dados")
            col1, col2 = st.columns([1, 1])

            with col1:
                self.render_csv_upload()

            with col2:
                self.render_csv_preview()

            # Separador visual entre seções
            st.divider()

            # Seção inferior: Chat ocupando toda a largura
            self.render_chat_section()

            # Seção de debug (se habilitada)
            if st.session_state.show_debug_info:
                self.render_debug_section()

            # Verificar e exibir erros críticos do sistema
            self.check_system_health()

        except Exception as e:
            self.logger.error(f"Erro na execução da interface: {str(e)}")
            st.error(f"Erro interno da aplicação: {str(e)}")
            if st.session_state.show_debug_info:
                st.code(traceback.format_exc())

    def render_csv_upload(self):
        """Renderiza apenas a área de upload de CSV."""
        st.subheader("📤 Upload de Dados")

        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=['csv'],
            help="Faça upload de um arquivo CSV para começar a análise",
            key="csv_uploader"
        )

        if uploaded_file is not None:
            self.handle_csv_upload(uploaded_file)

        # Informações básicas se já houver arquivo carregado
        if st.session_state.csv_data is not None:
            st.success(f"✅ Arquivo carregado: **{st.session_state.csv_filename}**")
            data = st.session_state.csv_data
            st.info(f"📊 **{len(data):,}** linhas × **{len(data.columns)}** colunas")
        else:
            st.info("💡 Carregue um arquivo CSV para começar")

    def handle_csv_upload(self, uploaded_file):
        """Processa o upload do arquivo CSV."""
        try:
            # Ler o arquivo
            string_data = uploaded_file.read().decode('utf-8')
            csv_data = pd.read_csv(io.StringIO(string_data))

            # Validar os dados
            validator = CSVValidator()
            validation_result = validator.validate_csv_data(csv_data)

            if validation_result.get('is_valid', False):
                st.session_state.csv_data = csv_data
                st.session_state.csv_filename = uploaded_file.name
                st.session_state.csv_metadata = {
                    'filename': uploaded_file.name,
                    'rows': len(csv_data),
                    'columns': len(csv_data.columns),
                    'column_names': list(csv_data.columns),
                    'dtypes': csv_data.dtypes.astype(str).to_dict(),
                    'upload_time': datetime.now().isoformat()
                }
                self.logger.info(f"CSV carregado: {uploaded_file.name} ({len(csv_data)} linhas)")
            else:
                st.error("❌ Arquivo CSV inválido")
                for issue in validation_result.get('issues', []):
                    st.error(f"• {issue}")

        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
            self.logger.error(f"Erro no upload de CSV: {str(e)}")

    def render_csv_preview(self):
        """Renderiza preview dos dados CSV carregados."""
        st.subheader("📅 Preview dos Dados")

        if st.session_state.csv_data is not None:
            data = st.session_state.csv_data

            # Preview das primeiras linhas (sempre visível mas compacto)
            with st.expander("📋 Ver dados (primeiras 10 linhas)", expanded=True):
                st.dataframe(
                    sanitize_dataframe_for_arrow(data.head(10)),
                    use_container_width=True,
                    height=300
                )

            # Informações das colunas
            with st.expander("ℹ️ Informações das colunas", expanded=False):
                col_info = pd.DataFrame({
                    'Tipo': data.dtypes.astype(str),
                    'Não-Nulos': data.count(),
                    'Nulos': len(data) - data.count(),
                    'Valores Únicos': data.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
        else:
            st.info("📂 Nenhum arquivo carregado ainda")

    def render_chat_section(self):
        """Renderiza a seção de chat para consultas EDA ocupando toda a largura."""
        st.markdown("### 💬 Chat EDA")

        # Verificar se há dados carregados
        if st.session_state.csv_data is None:
            st.info("👆 Faça upload de um arquivo CSV na seção acima para começar a fazer perguntas sobre seus dados")
            return

        # Campo de entrada para consulta (mais amplo e visível)
        with st.form("query_form", clear_on_submit=True):
            user_query = st.text_area(
                "Faça sua pergunta sobre os dados:",
                placeholder="Ex: Quais são as estatísticas descritivas dos dados? Há outliers? Existe correlação entre as variáveis?",
                height=120,
                help="Digite sua pergunta em linguagem natural. Seja específico sobre que tipo de análise deseja."
            )

            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                submit_button = st.form_submit_button("🚀 Analisar", type="primary", use_container_width=True)
            with col2:
                st.markdown("_💡 Dica: Seja específico sobre que tipo de análise deseja realizar_")

        # Processar consulta se submetida
        if submit_button and user_query:
            self.process_user_query(user_query)

        # Separador antes do histórico
        if st.session_state.chat_history:
            st.divider()

        # Container para histórico de chat (largura total)
        chat_container = st.container()
        with chat_container:
            self.render_chat_history()

    def process_user_query(self, query: str):
        """Processa uma consulta do usuário."""
        if not query.strip():
            st.error("Por favor, digite uma pergunta válida.")
            return

        # Adicionar consulta ao histórico imediatamente
        chat_item = {
            "id": len(st.session_state.chat_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "user_query": query,
            "status": "processing",
            "response": None,
            "error": None,
            "execution_time": None
        }

        st.session_state.chat_history.append(chat_item)

        # Verificar cache primeiro
        cache_key = self.create_cache_key(query, st.session_state.csv_filename)
        cached_result = st.session_state.result_cache.get(cache_key)

        if cached_result and not st.session_state.show_debug_info:
            # Usar resultado em cache
            chat_item.update({
                "status": "completed",
                "response": cached_result["response"],
                "execution_time": cached_result["execution_time"],
                "from_cache": True
            })
            st.info("📋 Resultado obtido do cache")
            st.rerun()
            return

        # Mostrar status de processamento
        with st.spinner("🔄 Analisando seus dados..."):
            start_time = datetime.now()

            try:
                # Converter DataFrame para string CSV para o workflow
                csv_string = None
                if st.session_state.csv_data is not None:
                    csv_buffer = io.StringIO()
                    st.session_state.csv_data.to_csv(csv_buffer, index=False)
                    csv_string = csv_buffer.getvalue()

                # Executar o workflow EDA com session_id para memória contextual
                workflow_result = execute_eda_analysis_sync(
                    user_query=query,
                    csv_data=csv_string,
                    config={
                        "debug": st.session_state.show_debug_info,
                        "recursion_limit": 30,
                        "session_id": st.session_state.session_id  # Passa session_id para memória contextual
                    }
                )

                # Processar resultado do workflow
                if workflow_result.get("success", False):
                    # Formatar visualizações para Streamlit
                    visualizations = []
                    if workflow_result.get("visualizations"):
                        visualizations = self.format_visualizations_for_streamlit(
                            workflow_result["visualizations"]
                        )

                    # Resultado real do workflow
                    response = {
                        "success": True,
                        "user_query": query,
                        "query_classification": workflow_result.get("query_classification"),
                        "analysis_results": workflow_result.get("analysis_results", {}),
                        "generated_code": workflow_result.get("generated_code"),
                        "execution_results": workflow_result.get("execution_results"),
                        "visualizations": visualizations,
                        "final_insights": workflow_result.get("final_insights", "Análise concluída com sucesso."),
                        "response_data": workflow_result.get("response_data", {})
                    }
                else:
                    # Workflow falhou
                    response = {
                        "success": False,
                        "user_query": query,
                        "errors": workflow_result.get("errors", ["Erro desconhecido no workflow"]),
                        "partial_results": workflow_result.get("partial_results", {})
                    }

                execution_time = (datetime.now() - start_time).total_seconds()

                # Atualizar item do chat com resultado
                chat_item.update({
                    "status": "completed",
                    "response": response,
                    "execution_time": execution_time
                })

                # Salvar no cache se foi bem-sucedido
                if response.get("success", False):
                    st.session_state.result_cache[cache_key] = {
                        "response": response,
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat()
                    }

                st.success(f"✅ Análise concluída em {execution_time:.1f}s")

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = str(e)

                # Atualizar item do chat com erro
                chat_item.update({
                    "status": "error",
                    "error": error_msg,
                    "execution_time": execution_time
                })

                st.error(f"❌ Erro na análise: {error_msg}")
                self.logger.error(f"Erro no processamento da consulta: {error_msg}")

        # Forçar atualização da interface
        st.rerun()

    def render_chat_history(self):
        """Renderiza o histórico de conversas."""
        if not st.session_state.chat_history:
            st.info("💬 Nenhuma conversa ainda. Faça sua primeira pergunta!")
            return

        # Renderizar itens em ordem cronológica reversa (mais recente primeiro)
        for i, chat_item in enumerate(reversed(st.session_state.chat_history)):
            self.render_chat_item(chat_item, len(st.session_state.chat_history) - i)

    def render_chat_item(self, chat_item: Dict[str, Any], index: int):
        """Renderiza um item individual do chat com melhor aproveitamento horizontal."""
        timestamp = datetime.fromisoformat(chat_item["timestamp"])
        time_str = timestamp.strftime("%H:%M:%S")

        # Container para o item de chat
        with st.container():
            # Cabeçalho da pergunta com melhor layout
            col_q1, col_q2 = st.columns([6, 1])
            with col_q1:
                st.markdown(f"**🙋 Pergunta #{chat_item['id']}**")
                st.markdown(f"_{chat_item['user_query']}_")
            with col_q2:
                st.caption(f"🕒 {time_str}")

            # Status e resposta
            if chat_item["status"] == "processing":
                st.info("🔄 Processando...")

            elif chat_item["status"] == "completed":
                response = chat_item["response"]

                if response and response.get("success", False):
                    # Resposta bem-sucedida
                    execution_time = chat_item.get("execution_time", 0)
                    from_cache = chat_item.get("from_cache", False)
                    cache_indicator = " 📋" if from_cache else ""
                    st.success(f"✅ Análise concluída ({execution_time:.1f}s){cache_indicator}")

                    # Exibir insights principais (com mais espaço)
                    if response.get("final_insights"):
                        st.markdown("**📊 Resultados:**")
                        st.markdown(response["final_insights"])

                    # Exibir resposta formatada se disponível
                    response_data = response.get("response_data", {})
                    if response_data.get("formatted_response"):
                        with st.expander("📄 Resposta detalhada", expanded=False):
                            st.markdown(response_data["formatted_response"])

                    # Exibir visualizações se disponíveis (em layout mais amplo)
                    visualizations = response.get("visualizations", [])
                    if visualizations:
                        st.markdown("**📈 Visualizações:**")
                        # Se houver múltiplas visualizações, usar colunas quando apropriado
                        if len(visualizations) == 2:
                            viz_col1, viz_col2 = st.columns(2)
                            with viz_col1:
                                self.render_visualization(visualizations[0])
                            with viz_col2:
                                self.render_visualization(visualizations[1])
                        else:
                            for viz in visualizations:
                                self.render_visualization(viz)

                    # Exibir resultados de análise se disponíveis
                    analysis_results = response.get("analysis_results", {})
                    if analysis_results:
                        with st.expander("🔬 Dados da análise", expanded=False):
                            for analysis_type, results in analysis_results.items():
                                if analysis_type != "summary":
                                    st.markdown(f"**{analysis_type.title()}:**")
                                    if isinstance(results, dict):
                                        for key, value in results.items():
                                            st.text(f"  {key}: {value}")
                                    else:
                                        st.text(f"  {results}")

                else:
                    # Resposta com falha
                    st.error("❌ Falha na análise")
                    if "errors" in response:
                        for error in response["errors"]:
                            st.error(f"• {error}")

            elif chat_item["status"] == "error":
                # Erro no processamento
                st.error(f"❌ Erro: {chat_item.get('error', 'Erro desconhecido')}")

            # Separador entre itens (mais sutil)
            st.markdown("---")

    def render_visualization(self, visualization: Dict[str, Any]):
        """Renderiza uma visualização."""
        try:
            viz_type = visualization.get("type", "unknown")
            viz_data = visualization.get("data")

            if viz_type == "plotly" and viz_data:
                # Visualização Plotly
                st.plotly_chart(viz_data, use_container_width=True)
            elif viz_type == "matplotlib" and viz_data:
                # Visualização Matplotlib
                st.pyplot(viz_data)
            elif viz_type == "dataframe" and viz_data is not None:
                # Tabela de dados
                st.dataframe(sanitize_dataframe_for_arrow(viz_data), width="stretch")
            elif viz_type == "metric":
                # Métricas simples
                if isinstance(viz_data, dict):
                    cols = st.columns(len(viz_data))
                    for i, (key, value) in enumerate(viz_data.items()):
                        with cols[i]:
                            st.metric(key, value)
            elif viz_type == "text":
                # Texto formatado
                viz_title = visualization.get("title", "")
                if viz_title:
                    st.markdown(f"**{viz_title}**")
                st.markdown(str(viz_data))
            elif viz_type == "image_base64":
                # Imagem codificada em base64
                self._render_base64_image(viz_data, visualization.get("title", "Visualizacao"))
            else:
                st.warning(f"Tipo de visualizacao nao suportado: {viz_type}")

        except Exception as e:
            st.error(f"Erro ao renderizar visualização: {str(e)}")
            self.logger.error(f"Erro na renderização de visualização: {str(e)}")

    def _render_base64_image(self, image_base64: Optional[str], title: str = "Visualização"):
        """
        Renderiza imagem base64 com configurações otimizadas para exibição adequada.

        Args:
            image_base64: String base64 da imagem
            title: Título da visualização
        """
        try:
            import base64

            # Exibir título
            st.markdown(f"**{title}**")

            if not image_base64:
                st.warning("Imagem base64 nao disponivel")
                return

            # Decodificar imagem
            image_bytes = base64.b64decode(image_base64)

            # Renderizar com configurações otimizadas
            # Usar width fixo para garantir tamanho adequado
            st.image(
                image_bytes,
                use_container_width=True,  # Usa largura da coluna ao invés de container
                output_format="PNG"
            )

        except Exception as e:
            st.error(f"Erro ao renderizar imagem: {str(e)}")
            self.logger.error(f"Erro na renderização de imagem base64: {str(e)}")

    def format_visualizations_for_streamlit(self, workflow_visualizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formata visualizações do workflow para exibição no Streamlit.

        Args:
            workflow_visualizations: Lista de visualizações do workflow

        Returns:
            Lista de visualizações formatadas para Streamlit
        """
        formatted_visualizations = []

        for viz in workflow_visualizations:
            try:
                viz_type = viz.get("type", "unknown")
                viz_data = viz.get("data")
                viz_title = viz.get("title", "Visualização")

                # Converter diferentes tipos de visualização
                if viz_type in ["plot", "chart", "graph"]:
                    # Tentar criar visualização Plotly se temos dados estruturados
                    if isinstance(viz_data, dict):
                        if "x" in viz_data and "y" in viz_data:
                            # Gráfico de dispersão/linha
                            fig = px.scatter(
                                x=viz_data["x"],
                                y=viz_data["y"],
                                title=viz_title
                            )
                            formatted_visualizations.append({
                                "type": "plotly",
                                "data": fig,
                                "title": viz_title
                            })
                        elif "labels" in viz_data and "values" in viz_data:
                            # Gráfico de pizza/barras
                            fig = px.pie(
                                values=viz_data["values"],
                                names=viz_data["labels"],
                                title=viz_title
                            )
                            formatted_visualizations.append({
                                "type": "plotly",
                                "data": fig,
                                "title": viz_title
                            })

                elif viz_type == "dataframe":
                    # Tabela de dados
                    formatted_visualizations.append({
                        "type": "dataframe",
                        "data": viz_data,
                        "title": viz_title
                    })

                elif viz_type == "metrics":
                    # Métricas
                    formatted_visualizations.append({
                        "type": "metric",
                        "data": viz_data,
                        "title": viz_title
                    })

                elif viz_type == "text":
                    # Texto formatado
                    formatted_visualizations.append({
                        "type": "text",
                        "data": viz_data,
                        "title": viz_title
                    })

                elif viz_type in ["histogram", "boxplot", "correlation_heatmap", "bar_chart", "scatter", "scatter_plot", "heatmap", "time_series", "line_plot", "bar_plot", "distribution_plot"]:
                    # Visualizações com imagem base64 (do graph_generator)
                    image_base64 = viz.get("image_base64")
                    if image_base64:
                        formatted_visualizations.append({
                            "type": "image_base64",
                            "data": image_base64,
                            "title": viz_title,
                            "metadata": viz.get("metadata", {})
                        })
                    else:
                        # Se não tem base64, tentar renderizar como texto
                        formatted_visualizations.append({
                            "type": "text",
                            "data": f"Visualização do tipo '{viz_type}' sem imagem disponível",
                            "title": viz_title
                        })

                else:
                    # Tipo desconhecido - tentar renderizar como texto
                    formatted_visualizations.append({
                        "type": "text",
                        "data": f"Visualização do tipo '{viz_type}': {str(viz_data)[:200]}...",
                        "title": viz_title
                    })

            except Exception as e:
                self.logger.error(f"Erro ao formatar visualização: {str(e)}")
                # Adicionar visualização de erro
                formatted_visualizations.append({
                    "type": "text",
                    "data": f"Erro ao processar visualização: {str(e)}",
                    "title": "Erro"
                })

        return formatted_visualizations

    def create_cache_key(self, query: str, filename: Optional[str] = None) -> str:
        """
        Cria uma chave única para cache baseada na consulta e arquivo.

        Args:
            query: Consulta do usuário
            filename: Nome do arquivo CSV (opcional)

        Returns:
            Chave única para cache
        """
        import hashlib

        # Normalizar consulta
        normalized_query = query.lower().strip()

        # Criar string única combinando consulta e arquivo
        cache_string = f"{normalized_query}|{filename or 'no_file'}"

        # Gerar hash MD5
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()[:16]

        return cache_key

    def check_system_health(self):
        """Verifica a saúde do sistema e exibe alertas se necessário."""
        try:
            # Verificar se há muitos erros recentes
            recent_errors = [
                item for item in st.session_state.chat_history
                if item.get("status") == "error"
                and (datetime.now() - datetime.fromisoformat(item["timestamp"])).total_seconds() < 300  # últimos 5 min
            ]

            if len(recent_errors) >= 3:
                st.error("⚠️ Múltiplos erros detectados nos últimos 5 minutos. Considere recarregar a página ou verificar seus dados.")

            # Verificar uso de memória do cache
            if len(st.session_state.result_cache) > 20:
                st.warning("💾 Cache ficando grande. Considere limpar o histórico para melhor performance.")

            # Verificar se a sessão está muito longa
            start_time = datetime.fromisoformat(st.session_state.session_start_time)
            session_duration = (datetime.now() - start_time).total_seconds()
            if session_duration > 3600:  # 1 hora
                st.info("⏰ Sessão longa detectada. Considere recarregar a página para evitar problemas de performance.")

        except Exception as e:
            self.logger.error(f"Erro na verificação de saúde do sistema: {str(e)}")

    def render_debug_section(self):
        """Renderiza seção de informações de debug."""
        if st.session_state.show_debug_info:
            st.divider()
            st.subheader("🔧 Debug Information")

            with st.expander("Session State", expanded=False):
                debug_state = {
                    key: str(value)[:200] + "..." if len(str(value)) > 200 else value
                    for key, value in st.session_state.items()
                    if not key.startswith('_')
                }
                st.json(debug_state)


def main():
    """Função principal da aplicação."""
    try:
        app = StreamlitEDAInterface()
        app.run()
    except Exception as e:
        st.error(f"Erro fatal na aplicação: {str(e)}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()