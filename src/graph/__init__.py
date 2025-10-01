"""
Modulo graph - Sistema de workflow LangGraph para analise EDA.

Este modulo implementa o sistema completo de workflow baseado em LangGraph
para coordenar a execucao de agentes especializados em analise exploratoria de dados.

Componentes principais:
- EDAState: Gerenciamento de estado do workflow
- Nodes: Implementacao dos nos de processamento
- Edges: Logica de roteamento e transicoes
- Workflow: Orquestrador principal do sistema

Uso basico:
    from src.graph import execute_eda_analysis_sync

    result = execute_eda_analysis_sync(
        user_query="Analise os dados de vendas",
        csv_data=csv_content
    )
"""

from .state import EDAState
from .workflow import (
    EDAWorkflow,
    EDAWorkflowError,
    get_workflow,
    reset_workflow,
    execute_eda_analysis,
    execute_eda_analysis_sync
)
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
    error_handler_node
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
    validate_state_transition,
    get_analysis_type_for_agent,
    get_next_node_for_analysis_type,
    EdgeError
)

# Exportacoes principais para uso externo
__all__ = [
    # Estado
    "EDAState",

    # Workflow principal
    "EDAWorkflow",
    "EDAWorkflowError",
    "get_workflow",
    "reset_workflow",
    "execute_eda_analysis",
    "execute_eda_analysis_sync",

    # Nos do workflow
    "entry_point_node",
    "classify_query_node",
    "process_data_node",
    "route_to_analysis_node",
    "descriptive_analysis_node",
    "pattern_detection_node",
    "anomaly_detection_node",
    "relationship_analysis_node",
    "code_generation_node",
    "execute_code_node",
    "create_visualization_node",
    "synthesize_results_node",
    "format_response_node",
    "error_handler_node",

    # Logica de arestas
    "classify_query_edge",
    "process_data_edge",
    "route_to_analysis_edge",
    "analysis_completion_edge",
    "code_generation_edge",
    "code_execution_edge",
    "visualization_edge",
    "synthesis_edge",
    "format_response_edge",
    "error_handler_edge",
    "validate_state_transition",
    "get_analysis_type_for_agent",
    "get_next_node_for_analysis_type",
    "EdgeError"
]

# Versao do modulo graph
__version__ = "0.0.1"

# Configuracao padrao para o workflow
DEFAULT_WORKFLOW_CONFIG = {
    "recursion_limit": 50,
    "debug": False,
    "enable_logging": True,
    "max_retries": 3
}


def validate_graph_module() -> bool:
    """
    Valida se todos os componentes do modulo graph estao funcionais.

    Returns:
        True se todos os componentes estao OK
    """
    try:
        # Testar importacao do workflow
        workflow = get_workflow()

        # Validar estrutura do workflow
        validation = workflow.validate_workflow()

        return validation.get("is_valid", False)

    except Exception:
        return False


def get_module_info() -> dict:
    """
    Retorna informacoes sobre o modulo graph.

    Returns:
        Dicionario com informacoes do modulo
    """
    try:
        workflow = get_workflow()
        structure = workflow.get_graph_structure()

        return {
            "version": __version__,
            "workflow_compiled": structure.get("is_compiled", False),
            "total_nodes": structure.get("total_nodes", 0),
            "entry_point": structure.get("entry_point"),
            "is_valid": validate_graph_module()
        }

    except Exception as e:
        return {
            "version": __version__,
            "error": str(e),
            "is_valid": False
        }