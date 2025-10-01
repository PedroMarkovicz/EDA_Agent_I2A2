"""
Orquestrador principal do workflow LangGraph para sistema EDA.
Define a estrutura completa do grafo, conecta agentes como nós,
estabelece fluxo de execução e coordena todo o processo de análise.
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END

from .state import EDAState
from .nodes import (
    entry_point_node,
    classify_query_node,
    process_data_node,
    route_to_analysis_node,
    descriptive_analysis_node,
    pattern_detection_node,
    anomaly_detection_node,
    relationship_analysis_node,
    code_generation_node,
    execute_code_node,
    create_visualization_node,
    synthesize_results_node,
    format_response_node,
    error_handler_node,
    tool_orchestration_node,
    tool_synthesis_node
)
from .edges import (
    classify_query_edge,
    process_data_edge,
    route_to_analysis_edge,
    analysis_completion_edge,
    code_generation_edge,
    code_execution_edge,
    visualization_edge,
    synthesis_edge,
    format_response_edge,
    error_handler_edge,
    validate_state_transition
)
from ..core.logger import get_logger


class EDAWorkflowError(Exception):
    """Erro específico do workflow EDA."""
    pass


class EDAWorkflow:
    """
    Orquestrador principal do workflow de análise exploratória de dados.

    Coordena a execução de agentes especializados através de um grafo LangGraph,
    gerenciando o fluxo de dados e garantindo a execução ordenada das análises.
    """

    def __init__(self, use_tool_based_workflow: bool = True):
        """
        Inicializa o workflow EDA.

        Args:
            use_tool_based_workflow: Se True, usa novo workflow baseado em ferramentas (recomendado).
                                    Se False, usa workflow antigo com agentes especializados.
        """
        self.logger = get_logger("workflow")
        self._graph: Optional[Any] = None
        self.use_tool_based_workflow = use_tool_based_workflow

        if use_tool_based_workflow:
            self._build_tool_based_graph()
        else:
            self._build_graph()

    def _build_graph(self) -> None:
        """
        Constrói o grafo LangGraph com todos os nós e arestas.
        """
        try:
            self.logger.info("Construindo grafo LangGraph para workflow EDA")

            # Criar o grafo com o estado EDA
            workflow = StateGraph(EDAState)

            # Adicionar todos os nós do workflow
            self._add_nodes(workflow)

            # Adicionar arestas e lógica de roteamento
            self._add_edges(workflow)

            # Definir ponto de entrada
            workflow.set_entry_point("entry_point")

            # Compilar o grafo
            self._graph = workflow.compile()

            self.logger.info("Grafo LangGraph construído com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao construir grafo: {str(e)}")
            raise EDAWorkflowError(f"Falha na construção do workflow: {str(e)}")

    def _add_nodes(self, workflow: StateGraph) -> None:
        """
        Adiciona todos os nós ao grafo.

        Args:
            workflow: Instância do StateGraph
        """
        nodes = {
            "entry_point": entry_point_node,
            "classify_query": classify_query_node,
            "process_data": process_data_node,
            "route_to_analysis": route_to_analysis_node,
            "descriptive_analysis": descriptive_analysis_node,
            "pattern_analysis": pattern_detection_node,
            "anomaly_analysis": anomaly_detection_node,
            "relationship_analysis": relationship_analysis_node,
            "generate_code": code_generation_node,
            "execute_code": execute_code_node,
            "create_visualizations": create_visualization_node,
            "synthesize_results": synthesize_results_node,
            "format_response": format_response_node,
            "error_handler": error_handler_node
        }

        for node_name, node_function in nodes.items():
            workflow.add_node(node_name, node_function)
            self.logger.debug(f"Nó adicionado: {node_name}")

    def _add_edges(self, workflow: StateGraph) -> None:
        """
        Adiciona arestas e lógica de roteamento ao grafo.

        Args:
            workflow: Instância do StateGraph
        """
        # Arestas fixas simples
        workflow.add_edge("entry_point", "classify_query")

        # Arestas condicionais com lógica de roteamento
        workflow.add_conditional_edges(
            "classify_query",
            classify_query_edge,
            {
                "process_data": "process_data",
                "route_to_analysis": "route_to_analysis",
                "error_handler": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "process_data",
            process_data_edge,
            {
                "route_to_analysis": "route_to_analysis",
                "error_handler": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "route_to_analysis",
            route_to_analysis_edge,
            {
                "descriptive_analysis": "descriptive_analysis",
                "pattern_analysis": "pattern_analysis",
                "anomaly_analysis": "anomaly_analysis",
                "relationship_analysis": "relationship_analysis",
                "generate_code": "generate_code",
                "synthesize_results": "synthesize_results",
                "error_handler": "error_handler"
            }
        )

        # Arestas para nós de análise (todos usam a mesma lógica)
        analysis_nodes = [
            "descriptive_analysis",
            "pattern_analysis",
            "anomaly_analysis",
            "relationship_analysis"
        ]

        for node in analysis_nodes:
            workflow.add_conditional_edges(
                node,
                analysis_completion_edge,
                {
                    "route_to_analysis": "route_to_analysis",
                    "generate_code": "generate_code",
                    "error_handler": "error_handler"
                }
            )

        workflow.add_conditional_edges(
            "generate_code",
            code_generation_edge,
            {
                "execute_code": "execute_code",
                "synthesize_results": "synthesize_results",
                "error_handler": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "execute_code",
            code_execution_edge,
            {
                "create_visualizations": "create_visualizations",
                "error_handler": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "create_visualizations",
            visualization_edge,
            {
                "synthesize_results": "synthesize_results",
                "error_handler": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "synthesize_results",
            synthesis_edge,
            {
                "format_response": "format_response",
                "error_handler": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "format_response",
            format_response_edge,
            {
                "END": END,
                "error_handler": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "error_handler",
            error_handler_edge,
            {
                "END": END
            }
        )

        self.logger.debug("Todas as arestas adicionadas ao grafo")

    async def execute(
        self,
        user_query: str,
        csv_data: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o workflow completo de análise EDA.

        Args:
            user_query: Consulta do usuário
            csv_data: Dados CSV para análise (opcional)
            config: Configurações adicionais (opcional)

        Returns:
            Resultado da análise EDA

        Raises:
            EDAWorkflowError: Se houver erro na execução
        """
        if not self._graph:
            raise EDAWorkflowError("Grafo não foi construído adequadamente")

        try:
            self.logger.info(f"Iniciando execução do workflow EDA para consulta: {user_query[:100]}...")

            # Extrair session_id do config se existir
            session_id = config.get("session_id") if config else None

            # Criar estado inicial com session_id
            initial_state = EDAState(
                user_query=user_query,
                csv_data=csv_data,
                session_id=session_id  # Propagar session_id para o estado
            )

            # Configuração padrão
            execution_config = {
                "recursion_limit": 50,
                "debug": False
            }
            if config:
                execution_config.update(config)

            # Executar o workflow
            result = await self._graph.ainvoke(
                initial_state,
                config=execution_config
            )

            # Processar resultado
            return self._process_result(result)

        except Exception as e:
            self.logger.error(f"Erro na execução do workflow: {str(e)}")
            raise EDAWorkflowError(f"Falha na execução: {str(e)}")

    def execute_sync(
        self,
        user_query: str,
        csv_data: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa o workflow de forma síncrona.

        Args:
            user_query: Consulta do usuário
            csv_data: Dados CSV para análise (opcional)
            config: Configurações adicionais (opcional)

        Returns:
            Resultado da análise EDA

        Raises:
            EDAWorkflowError: Se houver erro na execução
        """
        if not self._graph:
            raise EDAWorkflowError("Grafo não foi construído adequadamente")

        try:
            self.logger.info(f"Iniciando execução síncrona do workflow EDA")

            # Extrair session_id do config se existir
            session_id = config.get("session_id") if config else None

            # Criar estado inicial com session_id
            initial_state = EDAState(
                user_query=user_query,
                csv_data=csv_data,
                session_id=session_id  # Propagar session_id para o estado
            )

            # Configuração padrão
            execution_config = {
                "recursion_limit": 50,
                "debug": False
            }
            if config:
                execution_config.update(config)

            # Executar o workflow
            result = self._graph.invoke(
                initial_state,
                config=execution_config
            )

            # Processar resultado
            return self._process_result(result)

        except Exception as e:
            self.logger.error(f"Erro na execução síncrona do workflow: {str(e)}")
            raise EDAWorkflowError(f"Falha na execução síncrona: {str(e)}")

    def _process_result(self, result) -> Dict[str, Any]:
        """
        Processa o resultado final do workflow.

        Args:
            result: Estado final do workflow (EDAState ou dict)

        Returns:
            Dicionário com resultado processado
        """
        try:
            # Verificar se é dict ou EDAState
            if isinstance(result, dict):
                # Se é dict, converter ou acessar propriedades diretamente
                has_errors = result.get('errors', []) != []
                errors = result.get('errors', [])
            else:
                # Se é EDAState, usar métodos do objeto
                has_errors = result.has_errors()
                errors = result.errors

            # Verificar se houve erros
            if has_errors:
                self.logger.warning(f"Workflow finalizado com erros: {errors}")
                return {
                    "success": False,
                    "errors": errors,
                    "partial_results": result.get('analysis_results', {}) if isinstance(result, dict) else result.get_all_results()
                }

            # Resultado bem-sucedido
            if isinstance(result, dict):
                processed_result = {
                    "success": True,
                    "user_query": result.get('user_query', ''),
                    "query_classification": result.get('query_classification', {}),
                    "csv_metadata": result.get('csv_metadata', {}),
                    "analysis_results": result.get('analysis_results', {}),
                    "generated_code": result.get('generated_code', ''),
                    "execution_results": result.get('execution_results', {}),
                    "visualizations": result.get('visualizations', []),
                    "final_insights": result.get('final_insights', ''),
                    "response_data": result.get('response_data', {}),
                    "execution_time": result.get('execution_time', None)
                }
            else:
                processed_result = {
                    "success": True,
                    "user_query": result.user_query,
                    "query_classification": result.query_classification,
                    "csv_metadata": result.csv_metadata,
                    "analysis_results": result.get_all_results(),
                    "generated_code": result.generated_code,
                    "execution_results": result.execution_results,
                    "visualizations": result.visualizations,
                    "final_insights": result.final_insights,
                    "response_data": result.response_data,
                    "execution_time": getattr(result, 'execution_time', None)
                }

            self.logger.info("Workflow executado com sucesso")
            return processed_result

        except Exception as e:
            self.logger.error(f"Erro no processamento do resultado: {str(e)}")
            return {
                "success": False,
                "errors": [f"Erro no processamento: {str(e)}"],
                "partial_results": {}
            }

    def get_graph_structure(self) -> Dict[str, Any]:
        """
        Retorna informações sobre a estrutura do grafo.

        Returns:
            Dicionário com informações estruturais do grafo
        """
        try:
            nodes = [
                "entry_point", "classify_query", "process_data", "route_to_analysis",
                "descriptive_analysis", "pattern_analysis", "anomaly_analysis",
                "relationship_analysis", "generate_code", "execute_code",
                "create_visualizations", "synthesize_results", "format_response",
                "error_handler"
            ]

            edges = {
                "entry_point": ["classify_query"],
                "classify_query": ["process_data", "route_to_analysis", "error_handler"],
                "process_data": ["route_to_analysis", "error_handler"],
                "route_to_analysis": ["descriptive_analysis", "pattern_analysis",
                                    "anomaly_analysis", "relationship_analysis",
                                    "generate_code", "synthesize_results", "error_handler"],
                "descriptive_analysis": ["route_to_analysis", "generate_code", "error_handler"],
                "pattern_analysis": ["route_to_analysis", "generate_code", "error_handler"],
                "anomaly_analysis": ["route_to_analysis", "generate_code", "error_handler"],
                "relationship_analysis": ["route_to_analysis", "generate_code", "error_handler"],
                "generate_code": ["execute_code", "synthesize_results", "error_handler"],
                "execute_code": ["create_visualizations", "error_handler"],
                "create_visualizations": ["synthesize_results", "error_handler"],
                "synthesize_results": ["format_response", "error_handler"],
                "format_response": ["END", "error_handler"],
                "error_handler": ["END"]
            }

            return {
                "nodes": nodes,
                "edges": edges,
                "entry_point": "entry_point",
                "end_points": ["END"],
                "total_nodes": len(nodes),
                "is_compiled": self._graph is not None
            }

        except Exception as e:
            self.logger.error(f"Erro ao obter estrutura do grafo: {str(e)}")
            return {"error": str(e)}

    def validate_workflow(self) -> Dict[str, Any]:
        """
        Valida a integridade do workflow.

        Returns:
            Resultado da validação
        """
        try:
            validation_result = {
                "is_valid": True,
                "issues": [],
                "warnings": []
            }

            # Verificar se o grafo foi compilado
            if not self._graph:
                validation_result["is_valid"] = False
                validation_result["issues"].append("Grafo não foi compilado")

            # Verificar consistência das arestas
            structure = self.get_graph_structure()
            if "error" in structure:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Erro na estrutura: {structure['error']}")

            # Validações adicionais podem ser adicionadas aqui

            return validation_result

        except Exception as e:
            return {
                "is_valid": False,
                "issues": [f"Erro na validação: {str(e)}"],
                "warnings": []
            }

    def _build_tool_based_graph(self) -> None:
        """
        Constrói o novo grafo simplificado baseado em ferramentas estatísticas.

        Este workflow é mais simples, rápido e escalável:
        entry_point → process_data → tool_orchestration → tool_synthesis → format_response → END
        """
        try:
            self.logger.info("Construindo grafo LangGraph TOOL-BASED para workflow EDA")

            # Criar o grafo com o estado EDA
            workflow = StateGraph(EDAState)

            # Adicionar nós do workflow simplificado
            workflow.add_node("entry_point", entry_point_node)
            workflow.add_node("process_data", process_data_node)
            workflow.add_node("tool_orchestration", tool_orchestration_node)
            workflow.add_node("tool_synthesis", tool_synthesis_node)
            workflow.add_node("format_response", format_response_node)
            workflow.add_node("error_handler", error_handler_node)

            self.logger.info("Nós do workflow tool-based adicionados")

            # Definir fluxo simplificado
            workflow.set_entry_point("entry_point")

            # Fluxo linear principal
            workflow.add_edge("entry_point", "process_data")

            # Após processar dados, ir para orquestração de ferramentas
            def after_process_data(state: EDAState) -> str:
                if state.has_errors():
                    return "error_handler"
                return "tool_orchestration"

            workflow.add_conditional_edges(
                "process_data",
                after_process_data,
                {
                    "tool_orchestration": "tool_orchestration",
                    "error_handler": "error_handler"
                }
            )

            # Após orquestração, ir para síntese
            def after_tool_orchestration(state: EDAState) -> str:
                if state.has_errors():
                    return "error_handler"
                return "tool_synthesis"

            workflow.add_conditional_edges(
                "tool_orchestration",
                after_tool_orchestration,
                {
                    "tool_synthesis": "tool_synthesis",
                    "error_handler": "error_handler"
                }
            )

            # Após síntese, formatar resposta
            def after_tool_synthesis(state: EDAState) -> str:
                if state.has_errors():
                    return "error_handler"
                return "format_response"

            workflow.add_conditional_edges(
                "tool_synthesis",
                after_tool_synthesis,
                {
                    "format_response": "format_response",
                    "error_handler": "error_handler"
                }
            )

            # Após formatação, finalizar
            workflow.add_edge("format_response", END)
            workflow.add_edge("error_handler", END)

            # Compilar o grafo
            self._graph = workflow.compile()

            self.logger.info("Grafo LangGraph TOOL-BASED construído com sucesso (4 nós principais)")

        except Exception as e:
            self.logger.error(f"Erro ao construir grafo tool-based: {str(e)}")
            raise EDAWorkflowError(f"Falha na construção do workflow tool-based: {str(e)}")


# Instância global do workflow (singleton)
_workflow_instance: Optional[EDAWorkflow] = None


def get_workflow() -> EDAWorkflow:
    """
    Retorna a instância singleton do workflow EDA.

    Returns:
        Instância do EDAWorkflow
    """
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = EDAWorkflow()
    return _workflow_instance


def reset_workflow() -> None:
    """
    Reseta a instância singleton do workflow.
    Útil para testes e reinicializações.
    """
    global _workflow_instance
    _workflow_instance = None


# Funções de conveniência para execução direta
async def execute_eda_analysis(
    user_query: str,
    csv_data: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Função de conveniência para executar análise EDA de forma assíncrona.

    Args:
        user_query: Consulta do usuário
        csv_data: Dados CSV para análise (opcional)
        config: Configurações adicionais (opcional)

    Returns:
        Resultado da análise EDA
    """
    workflow = get_workflow()
    return await workflow.execute(user_query, csv_data, config)


def execute_eda_analysis_sync(
    user_query: str,
    csv_data: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Função de conveniência para executar análise EDA de forma síncrona.

    Args:
        user_query: Consulta do usuário
        csv_data: Dados CSV para análise (opcional)
        config: Configurações adicionais (opcional)

    Returns:
        Resultado da análise EDA
    """
    workflow = get_workflow()
    return workflow.execute_sync(user_query, csv_data, config)