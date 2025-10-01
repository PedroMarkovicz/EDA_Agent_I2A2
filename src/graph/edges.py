"""
Definições das conexões e lógica de transição entre nós do grafo.
Implementa arestas condicionais, roteamento dinâmico e regras de fluxo
que determinam como os dados transitam pelo sistema multi-agente.
"""

from typing import Dict, List, Any, Optional, Union
from ..models.enums import EDAAnalysisType
from .state import EDAState
from ..core.logger import get_logger


class EdgeError(Exception):
    """Erro específico do sistema de arestas."""
    pass


def classify_query_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após a classificação da consulta.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros na classificação
        if state.has_errors():
            logger.warning("Erros detectados na classificação, direcionando para tratamento de erro")
            return "error_handler"

        # Verificar se a classificação foi realizada
        if not state.query_classification or not state.required_agents:
            logger.error("Classificação da consulta não foi realizada adequadamente")
            state.add_error("Falha na classificação da consulta")
            return "error_handler"

        # Se há dados CSV para processar, ir para processamento de dados
        if state.csv_data is not None:
            logger.info("CSV detectado, direcionando para processamento de dados")
            return "process_data"

        # Se não há dados CSV, mas há consulta classificada, tentar prosseguir
        if state.processed_query:
            logger.info("Consulta processada sem CSV, direcionando para análise")
            return "route_to_analysis"

        # Fallback para tratamento de erro
        logger.error("Estado inconsistente após classificação")
        state.add_error("Estado inconsistente após classificação da consulta")
        return "error_handler"

    except Exception as e:
        logger.error(f"Erro na aresta de classificação: {str(e)}")
        state.add_error(f"Erro no roteamento após classificação: {str(e)}")
        return "error_handler"


def process_data_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após o processamento de dados.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros no processamento
        if state.has_errors():
            logger.warning("Erros detectados no processamento, direcionando para tratamento")
            return "error_handler"

        # Verificar se os dados foram processados adequadamente
        if state.csv_data is None or state.csv_metadata is None:
            logger.error("Dados não foram processados adequadamente")
            state.add_error("Falha no processamento dos dados CSV")
            return "error_handler"

        # Direcionar para roteamento de análise
        logger.info("Dados processados com sucesso, direcionando para análise")
        return "route_to_analysis"

    except Exception as e:
        logger.error(f"Erro na aresta de processamento: {str(e)}")
        state.add_error(f"Erro no roteamento após processamento: {str(e)}")
        return "error_handler"


def route_to_analysis_edge(state: EDAState) -> str:
    """
    Roteia para o primeiro tipo de análise necessário.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó de análise
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros
        if state.has_errors():
            return "error_handler"

        # Verificar se há agentes para executar
        if not state.required_agents:
            logger.warning("Nenhum agente necessário identificado, direcionando para síntese")
            return "synthesize_results"

        # Determinar qual análise executar baseado nos agentes necessários e no que já foi executado
        analysis_priority = [
            (EDAAnalysisType.DESCRIPTIVE, "descriptive_analysis"),
            (EDAAnalysisType.PATTERN, "pattern_analysis"),
            (EDAAnalysisType.ANOMALY, "anomaly_analysis"),
            (EDAAnalysisType.RELATIONSHIP, "relationship_analysis")
        ]

        for analysis_type, analysis_node in analysis_priority:
            if (analysis_type.value in state.required_agents and
                not state.is_analysis_completed(analysis_type)):
                logger.info(f"Direcionando para análise: {analysis_node}")
                return analysis_node

        # Se todas as análises necessárias foram completadas, ir para geração de código
        logger.info("Todas as análises necessárias completadas, direcionando para geração de código")
        return "generate_code"

    except Exception as e:
        logger.error(f"Erro no roteamento para análise: {str(e)}")
        state.add_error(f"Erro no roteamento para análise: {str(e)}")
        return "error_handler"


def analysis_completion_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após completar uma análise.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros
        if state.has_errors():
            return "error_handler"

        # Verificar se ainda há análises pendentes
        pending_analyses = []
        analysis_mapping = [
            (EDAAnalysisType.DESCRIPTIVE, "data_analyzer"),
            (EDAAnalysisType.PATTERN, "pattern_detector"),
            (EDAAnalysisType.ANOMALY, "anomaly_detector"),
            (EDAAnalysisType.RELATIONSHIP, "relationship_analyzer")
        ]

        for analysis_type, agent_name in analysis_mapping:
            if (analysis_type.value in state.required_agents and
                not state.is_analysis_completed(analysis_type)):
                pending_analyses.append(agent_name)

        # Se há análises pendentes, continuar roteamento
        if pending_analyses:
            logger.info(f"Análises pendentes: {pending_analyses}, continuando roteamento")
            return "route_to_analysis"

        # Se todas as análises foram completadas, ir para geração de código
        logger.info("Todas as análises completadas, direcionando para geração de código")
        return "generate_code"

    except Exception as e:
        logger.error(f"Erro na verificação de completude de análise: {str(e)}")
        state.add_error(f"Erro na verificação de análises: {str(e)}")
        return "error_handler"


def code_generation_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após a geração de código.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros
        if state.has_errors():
            return "error_handler"

        # Verificar se código foi gerado
        if not state.generated_code:
            logger.warning("Nenhum código foi gerado, direcionando para síntese")
            return "synthesize_results"

        # Se código foi gerado, executar
        logger.info("Código gerado, direcionando para execução")
        return "execute_code"

    except Exception as e:
        logger.error(f"Erro na aresta de geração de código: {str(e)}")
        state.add_error(f"Erro após geração de código: {str(e)}")
        return "error_handler"


def code_execution_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após a execução de código.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros
        if state.has_errors():
            return "error_handler"

        # Sempre ir para criação de visualizações após execução
        logger.info("Código executado, direcionando para criação de visualizações")
        return "create_visualizations"

    except Exception as e:
        logger.error(f"Erro na aresta de execução de código: {str(e)}")
        state.add_error(f"Erro após execução de código: {str(e)}")
        return "error_handler"


def visualization_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após a criação de visualizações.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros
        if state.has_errors():
            return "error_handler"

        # Sempre ir para síntese após visualizações
        logger.info("Visualizações criadas, direcionando para síntese")
        return "synthesize_results"

    except Exception as e:
        logger.error(f"Erro na aresta de visualização: {str(e)}")
        state.add_error(f"Erro após criação de visualizações: {str(e)}")
        return "error_handler"


def synthesis_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após a síntese de resultados.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros
        if state.has_errors():
            return "error_handler"

        # Verificar se a síntese foi realizada
        if not state.final_insights:
            logger.warning("Síntese não foi realizada adequadamente")
            state.add_error("Falha na síntese de resultados")
            return "error_handler"

        # Ir para formatação da resposta final
        logger.info("Síntese completada, direcionando para formatação")
        return "format_response"

    except Exception as e:
        logger.error(f"Erro na aresta de síntese: {str(e)}")
        state.add_error(f"Erro após síntese: {str(e)}")
        return "error_handler"


def format_response_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após a formatação da resposta.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó (END para finalizar)
    """
    logger = get_logger("edges")

    try:
        # Verificar se há erros
        if state.has_errors():
            return "error_handler"

        # Verificar se a resposta foi formatada
        if not state.response_data:
            logger.warning("Resposta não foi formatada adequadamente")
            state.add_error("Falha na formatação da resposta")
            return "error_handler"

        # Workflow completado com sucesso
        logger.info("Resposta formatada com sucesso, finalizando workflow")
        return "END"

    except Exception as e:
        logger.error(f"Erro na aresta de formatação: {str(e)}")
        state.add_error(f"Erro na formatação da resposta: {str(e)}")
        return "error_handler"


def error_handler_edge(state: EDAState) -> str:
    """
    Determina o próximo nó após tratamento de erro.

    Args:
        state: Estado atual do workflow

    Returns:
        Nome do próximo nó (sempre END)
    """
    logger = get_logger("edges")

    try:
        # Log dos erros para debugging
        if state.errors:
            logger.error(f"Workflow finalizado com erros: {state.errors}")

        # Sempre finalizar após tratamento de erro
        return "END"

    except Exception as e:
        # Erro crítico no tratamento de erro
        logger.critical(f"Erro crítico no tratamento de erro: {str(e)}")
        return "END"


def should_continue_workflow(state: EDAState) -> bool:
    """
    Determina se o workflow deve continuar ou ser finalizado.

    Args:
        state: Estado atual do workflow

    Returns:
        True se deve continuar, False se deve finalizar
    """
    try:
        # Não continuar se há erros críticos
        if state.has_errors():
            return False

        # Não continuar se não há consulta
        if not state.user_query:
            return False

        # Continuar se ainda há processamento pendente
        return True

    except Exception:
        return False


def get_analysis_type_for_agent(agent_name: str) -> EDAAnalysisType:
    """
    Mapeia nome do agente para tipo de análise correspondente.

    Args:
        agent_name: Nome do agente

    Returns:
        Tipo de análise correspondente
    """
    mapping = {
        "data_analyzer": EDAAnalysisType.DESCRIPTIVE,
        "pattern_detector": EDAAnalysisType.PATTERN,
        "anomaly_detector": EDAAnalysisType.ANOMALY,
        "relationship_analyzer": EDAAnalysisType.RELATIONSHIP
    }

    return mapping.get(agent_name, EDAAnalysisType.DESCRIPTIVE)


def get_next_node_for_analysis_type(analysis_type: EDAAnalysisType) -> str:
    """
    Mapeia tipo de análise para nome do nó correspondente.

    Args:
        analysis_type: Tipo de análise

    Returns:
        Nome do nó correspondente
    """
    mapping = {
        EDAAnalysisType.DESCRIPTIVE: "descriptive_analysis",
        EDAAnalysisType.PATTERN: "pattern_analysis",
        EDAAnalysisType.ANOMALY: "anomaly_analysis",
        EDAAnalysisType.RELATIONSHIP: "relationship_analysis"
    }

    return mapping.get(analysis_type, "descriptive_analysis")


def validate_state_transition(current_node: str, next_node: str, state: EDAState) -> bool:
    """
    Valida se uma transição de estado é válida.

    Args:
        current_node: Nó atual
        next_node: Próximo nó
        state: Estado atual

    Returns:
        True se a transição é válida
    """
    logger = get_logger("edges")

    try:
        # Definir transições válidas
        valid_transitions = {
            "entry_point": ["classify_query", "error_handler"],
            "classify_query": ["process_data", "route_to_analysis", "error_handler"],
            "process_data": ["route_to_analysis", "error_handler"],
            "route_to_analysis": ["descriptive_analysis", "pattern_analysis",
                                 "anomaly_analysis", "relationship_analysis",
                                 "generate_code", "error_handler"],
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

        # Verificar se a transição é válida
        allowed_next = valid_transitions.get(current_node, [])
        is_valid = next_node in allowed_next

        if not is_valid:
            logger.warning(f"Transição inválida: {current_node} -> {next_node}")

        return is_valid

    except Exception as e:
        logger.error(f"Erro na validação de transição: {str(e)}")
        return False


# Mapeamento de funções de aresta por nó
EDGE_FUNCTIONS = {
    "classify_query": classify_query_edge,
    "process_data": process_data_edge,
    "route_to_analysis": route_to_analysis_edge,
    "descriptive_analysis": analysis_completion_edge,
    "pattern_analysis": analysis_completion_edge,
    "anomaly_analysis": analysis_completion_edge,
    "relationship_analysis": analysis_completion_edge,
    "generate_code": code_generation_edge,
    "execute_code": code_execution_edge,
    "create_visualizations": visualization_edge,
    "synthesize_results": synthesis_edge,
    "format_response": format_response_edge,
    "error_handler": error_handler_edge
}