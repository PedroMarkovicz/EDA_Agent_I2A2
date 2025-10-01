"""
Definições de todos os nós individuais do LangGraph.
Implementa função de cada nó no workflow, desde entrada de dados até
formatação final de respostas, incluindo nós de análise e execução.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from .state import EDAState
from ..core.query_interpreter import get_query_interpreter
from ..core.csv_processor import get_csv_processor
from ..core.logger import get_logger, log_data_operation, log_error_with_context
from ..agents import (
    get_data_analyzer_agent,
    get_pattern_detector_agent,
    get_anomaly_detector_agent,
    get_relationship_analyzer_agent,
    get_conclusion_generator_agent,
    get_code_generator_agent
)
from ..utils import (
    validate_dataframe,
    validate_user_query,
    get_graph_generator,
    get_response_formatter,
    execute_code_safely
)
from ..models.enums import EDAAnalysisType


def entry_point_node(state: EDAState) -> EDAState:
    """
    Nó de entrada do sistema - recebe consulta do usuário e dados CSV.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com dados de entrada processados
    """
    logger = get_logger("entry_point")

    try:
        log_data_operation(
            "entry_point_started",
            {
                "has_query": bool(state.user_query),
                "has_csv_data": state.csv_data is not None,
                "query_length": len(state.user_query) if state.user_query else 0
            },
            "entry_point"
        )

        # Validar entrada do usuário
        if not state.user_query.strip():
            state.add_error("Consulta do usuário está vazia")
            return state

        # Validar se há dados CSV
        if state.csv_data is None:
            state.add_error("Dados CSV não foram fornecidos")
            return state

        # Converter string CSV para DataFrame se necessário
        if isinstance(state.csv_data, str):
            try:
                import pandas as pd
                import io
                csv_buffer = io.StringIO(state.csv_data)
                state.csv_data = pd.read_csv(csv_buffer)
            except Exception as e:
                state.add_error(f"Erro ao processar dados CSV: {str(e)}")
                return state

        # Validar estrutura dos dados CSV
        validation_result = validate_dataframe(state.csv_data, "entry_point")
        if not validation_result["is_valid"]:
            state.add_error(f"Dados CSV inválidos: {validation_result['errors']}")
            return state

        # Armazenar metadados dos dados
        state.csv_metadata = {
            "shape": state.csv_data.shape,
            "columns": list(state.csv_data.columns),
            "dtypes": {col: str(dtype) for col, dtype in state.csv_data.dtypes.items()},
            "memory_usage_mb": state.csv_data.memory_usage(deep=True).sum() / (1024 * 1024),
            "validation_info": validation_result
        }

        # Registrar entrada no histórico de conversação
        state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "user_query",
            "content": state.user_query,
            "data_shape": state.csv_data.shape
        })

        log_data_operation(
            "entry_point_completed",
            {
                "csv_shape": state.csv_data.shape,
                "csv_columns": len(state.csv_data.columns),
                "has_errors": state.has_errors()
            },
            "entry_point"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"query_preview": state.user_query[:100] if state.user_query else ""},
            "entry_point_error"
        )
        state.add_error(f"Erro no ponto de entrada: {str(e)}")
        return state


def classify_query_node(state: EDAState) -> EDAState:
    """
    Nó de classificação - analisa consulta e determina tipos de análise necessários.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com classificação da consulta
    """
    logger = get_logger("classify_query")

    try:
        if state.has_errors():
            return state

        # Validar segurança da consulta
        query_validation = validate_user_query(state.user_query)
        if not query_validation["is_valid"]:
            state.add_error(f"Consulta insegura: {query_validation['errors']}")
            return state

        # Classificar consulta usando query interpreter
        query_interpreter = get_query_interpreter()
        classification_result = query_interpreter.interpret_query(
            state.user_query,
            state.csv_metadata
        )

        state.query_classification_result = classification_result
        state.query_classification = [agent.value for agent in classification_result.required_agents]
        state.required_agents = [agent.value for agent in classification_result.required_agents]

        # Registrar query na sessão de memória se session_id existir
        if state.session_id:
            try:
                from ..core.memory_manager import get_memory_manager
                memory_manager = get_memory_manager()

                # Criar UserQuery object
                from ..models.query_schema import UserQuery
                user_query_obj = UserQuery(
                    query_text=state.user_query,
                    intent_type=classification_result.intent_type,
                    analysis_types=classification_result.required_agents,
                    requires_visualization=classification_result.requires_visualization
                )

                # Salvar query e classificação na sessão
                memory_manager.save_query(
                    session_id=state.session_id,
                    query=user_query_obj,
                    classification=classification_result
                )
                logger.info(f"Query registrada na sessão de memória: {state.session_id}")
            except Exception as e:
                logger.warning(f"Erro ao registrar query na memória: {e}")

        log_data_operation(
            "query_classified",
            {
                "required_agents": [agent.value for agent in classification_result.required_agents],
                "confidence": classification_result.confidence_score,
                "processing_order": [agent.value for agent in classification_result.processing_order]
            },
            "classify_query"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"query_preview": state.user_query[:100]},
            "classify_query_error"
        )
        state.add_error(f"Erro na classificação da consulta: {str(e)}")
        return state


def process_data_node(state: EDAState) -> EDAState:
    """
    Nó de processamento de dados - validação e preparação dos dados CSV.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com dados processados
    """
    logger = get_logger("process_data")

    try:
        if state.has_errors():
            return state

        # Processar dados usando CSV processor
        csv_processor = get_csv_processor()

        # Validar estrutura completa
        validation_info = csv_processor.validate_csv_structure(state.csv_data)

        if validation_info.get("warnings"):
            for warning in validation_info["warnings"]:
                logger.warning(f"Aviso nos dados: {warning}")

        # Adicionar informações de processamento aos metadados
        state.csv_metadata.update({
            "processing_timestamp": datetime.now().isoformat(),
            "detailed_validation": validation_info,
            "is_ready_for_analysis": True
        })

        log_data_operation(
            "data_processed",
            {
                "validation_warnings": len(validation_info.get("warnings", [])),
                "is_ready": True
            },
            "process_data"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"data_shape": state.csv_data.shape if state.csv_data is not None else None},
            "process_data_error"
        )
        state.add_error(f"Erro no processamento dos dados: {str(e)}")
        return state


def descriptive_analysis_node(state: EDAState) -> EDAState:
    """
    Nó de análise descritiva - executa DataAnalyzerAgent.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resultados da análise descritiva
    """
    logger = get_logger("descriptive_analysis")

    try:
        if state.has_errors() or not state.should_execute_agent("data_analyzer"):
            return state

        # Executar análise descritiva
        data_analyzer = get_data_analyzer_agent()

        context = {
            "query": state.user_query,
            "metadata": state.csv_metadata,
            "analysis_timestamp": datetime.now().isoformat()
        }

        result = data_analyzer.analyze(state.csv_data, context)
        state.descriptive_results = result
        state.add_completed_analysis(EDAAnalysisType.DESCRIPTIVE)

        log_data_operation(
            "descriptive_analysis_completed",
            {
                "columns_analyzed": len(result.column_summaries) if result.column_summaries else 0,
                "has_overview": result.dataset_overview is not None
            },
            "descriptive_analysis"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"data_shape": state.csv_data.shape if state.csv_data is not None else None},
            "descriptive_analysis_error"
        )
        state.add_error(f"Erro na análise descritiva: {str(e)}")
        return state


def pattern_detection_node(state: EDAState) -> EDAState:
    """
    Nó de detecção de padrões - executa PatternDetectorAgent.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resultados da detecção de padrões
    """
    logger = get_logger("pattern_detection")

    try:
        if state.has_errors() or not state.should_execute_agent("pattern_detector"):
            return state

        # Executar detecção de padrões
        pattern_detector = get_pattern_detector_agent()

        context = {
            "query": state.user_query,
            "metadata": state.csv_metadata,
            "analysis_timestamp": datetime.now().isoformat()
        }

        result = pattern_detector.analyze(state.csv_data, context)
        state.pattern_results = result
        state.add_completed_analysis(EDAAnalysisType.PATTERN)

        log_data_operation(
            "pattern_detection_completed",
            {
                "temporal_patterns": len(result.temporal_patterns) if result.temporal_patterns else 0,
                "clusters_found": len(result.clusters) if result.clusters else 0
            },
            "pattern_detection"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"data_shape": state.csv_data.shape if state.csv_data is not None else None},
            "pattern_detection_error"
        )
        state.add_error(f"Erro na detecção de padrões: {str(e)}")
        return state


def anomaly_detection_node(state: EDAState) -> EDAState:
    """
    Nó de detecção de anomalias - executa AnomalyDetectorAgent.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resultados da detecção de anomalias
    """
    logger = get_logger("anomaly_detection")

    try:
        if state.has_errors() or not state.should_execute_agent("anomaly_detector"):
            return state

        # Executar detecção de anomalias
        anomaly_detector = get_anomaly_detector_agent()

        context = {
            "query": state.user_query,
            "metadata": state.csv_metadata,
            "analysis_timestamp": datetime.now().isoformat()
        }

        result = anomaly_detector.analyze(state.csv_data, context)
        state.anomaly_results = result
        state.add_completed_analysis(EDAAnalysisType.ANOMALY)

        outlier_count = sum(
            len(info.get("outliers", []))
            for info in result.outliers_by_column.values()
        ) if result.outliers_by_column else 0

        log_data_operation(
            "anomaly_detection_completed",
            {
                "columns_analyzed": len(result.outliers_by_column) if result.outliers_by_column else 0,
                "total_outliers": outlier_count
            },
            "anomaly_detection"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"data_shape": state.csv_data.shape if state.csv_data is not None else None},
            "anomaly_detection_error"
        )
        state.add_error(f"Erro na detecção de anomalias: {str(e)}")
        return state


def relationship_analysis_node(state: EDAState) -> EDAState:
    """
    Nó de análise de relacionamentos - executa RelationshipAnalyzerAgent.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resultados da análise de relacionamentos
    """
    logger = get_logger("relationship_analysis")

    try:
        if state.has_errors() or not state.should_execute_agent("relationship_analyzer"):
            return state

        # Executar análise de relacionamentos
        relationship_analyzer = get_relationship_analyzer_agent()

        context = {
            "query": state.user_query,
            "metadata": state.csv_metadata,
            "analysis_timestamp": datetime.now().isoformat()
        }

        result = relationship_analyzer.analyze(state.csv_data, context)
        state.relationship_results = result
        state.add_completed_analysis(EDAAnalysisType.RELATIONSHIP)

        log_data_operation(
            "relationship_analysis_completed",
            {
                "has_correlation_matrix": result.correlation_matrix is not None,
                "has_multicollinearity": result.multicollinearity_analysis is not None
            },
            "relationship_analysis"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"data_shape": state.csv_data.shape if state.csv_data is not None else None},
            "relationship_analysis_error"
        )
        state.add_error(f"Erro na análise de relacionamentos: {str(e)}")
        return state


def code_generation_node(state: EDAState) -> EDAState:
    """
    Nó de geração de código - executa CodeGeneratorAgent.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com código gerado
    """
    logger = get_logger("code_generation")

    try:
        if state.has_errors() or not state.should_execute_agent("code_generator"):
            return state

        # Executar geração de código
        code_generator = get_code_generator_agent()

        context = {
            "query": state.user_query,
            "metadata": state.csv_metadata,
            "available_results": state.get_all_results(),
            "analysis_timestamp": datetime.now().isoformat()
        }

        generated_code = code_generator.generate_analysis_code(
            state.csv_data,
            state.user_query,
            context
        )

        state.generated_code = generated_code

        log_data_operation(
            "code_generation_completed",
            {
                "code_length": len(generated_code),
                "has_code": bool(generated_code.strip())
            },
            "code_generation"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"query_preview": state.user_query[:100]},
            "code_generation_error"
        )
        state.add_error(f"Erro na geração de código: {str(e)}")
        return state


def execute_code_node(state: EDAState) -> EDAState:
    """
    Nó de execução de código - executa código Python gerado de forma segura.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resultados da execução
    """
    logger = get_logger("execute_code")

    try:
        if state.has_errors() or not state.generated_code.strip():
            return state

        # Preparar variáveis locais para execução
        local_vars = {
            "df": state.csv_data,
            "data": state.csv_data
        }

        # Executar código de forma segura
        execution_result = execute_code_safely(state.generated_code, local_vars)

        state.execution_results = execution_result

        if not execution_result["success"]:
            state.add_error(f"Erro na execução do código: {execution_result['error']}")

        log_data_operation(
            "code_execution_completed",
            {
                "success": execution_result["success"],
                "output_length": len(execution_result.get("output", "")),
                "variables_created": len(execution_result.get("variables_created", []))
            },
            "execute_code"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"code_preview": state.generated_code[:200]},
            "execute_code_error"
        )
        state.add_error(f"Erro na execução do código: {str(e)}")
        return state


def create_visualization_node(state: EDAState) -> EDAState:
    """
    Nó de criação de visualizações - gera gráficos usando GraphGenerator.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com visualizações criadas
    """
    logger = get_logger("create_visualization")

    try:
        if state.has_errors():
            return state

        graph_generator = get_graph_generator()
        visualizations = []

        # Gerar visualizações baseadas nos resultados das análises
        numeric_columns = state.csv_data.select_dtypes(include=['number']).columns

        # Histograma para primeira variável numérica
        if len(numeric_columns) > 0:
            hist_result = graph_generator.create_histogram(
                state.csv_data[numeric_columns[0]],
                title=f"Distribuição de {numeric_columns[0]}"
            )
            visualizations.append(hist_result)

        # Matriz de correlação se há múltiplas variáveis numéricas
        if len(numeric_columns) > 1:
            corr_result = graph_generator.create_correlation_heatmap(
                state.csv_data[numeric_columns],
                title="Matriz de Correlação"
            )
            visualizations.append(corr_result)

        # Boxplot para detecção de outliers
        if len(numeric_columns) > 0:
            box_result = graph_generator.create_boxplot(
                state.csv_data[numeric_columns[0]],
                title=f"Boxplot de {numeric_columns[0]}"
            )
            visualizations.append(box_result)

        state.visualizations = visualizations

        log_data_operation(
            "visualizations_created",
            {
                "count": len(visualizations),
                "types": [viz.get("type") for viz in visualizations]
            },
            "create_visualization"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"data_shape": state.csv_data.shape if state.csv_data is not None else None},
            "create_visualization_error"
        )
        state.add_error(f"Erro na criação de visualizações: {str(e)}")
        return state


def synthesize_results_node(state: EDAState) -> EDAState:
    """
    Nó de síntese - consolida resultados usando ConclusionGeneratorAgent.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com insights consolidados
    """
    logger = get_logger("synthesize_results")

    try:
        if state.has_errors() or not state.is_ready_for_synthesis():
            return state

        # Consolidar resultados usando conclusion generator
        conclusion_generator = get_conclusion_generator_agent()

        # Preparar todos os resultados para síntese
        all_results = state.get_all_results()

        context = {
            "query": state.user_query,
            "metadata": state.csv_metadata,
            "conversation_history": state.conversation_history,
            "analysis_timestamp": datetime.now().isoformat()
        }

        final_conclusion = conclusion_generator.consolidate_results(
            descriptive_result=state.descriptive_results,
            pattern_result=state.pattern_results,
            anomaly_result=state.anomaly_results,
            relationship_result=state.relationship_results,
            session_id=None,  # Pode ser passado se houver sessão
            context=context
        )

        state.final_insights = final_conclusion.response_text

        log_data_operation(
            "results_synthesized",
            {
                "response_length": len(final_conclusion.response_text),
                "all_insights_count": len(final_conclusion.all_insights),
                "priority_insights_count": len(final_conclusion.priority_insights)
            },
            "synthesize_results"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"available_results": list(state.get_all_results().keys())},
            "synthesize_results_error"
        )
        state.add_error(f"Erro na síntese de resultados: {str(e)}")
        return state


def format_response_node(state: EDAState) -> EDAState:
    """
    Nó de formatação - formata resposta final usando ResponseFormatter.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resposta formatada
    """
    logger = get_logger("format_response")

    try:
        formatter = get_response_formatter()

        # Formatar resposta final
        formatted_response = {
            "query": state.user_query,
            "timestamp": datetime.now().isoformat(),
            "data_summary": formatter.format_data_summary(state.csv_data),
            "analysis_results": {},
            "visualizations": state.visualizations,
            "insights": state.final_insights,
            "errors": state.errors if state.has_errors() else []
        }

        # Adicionar resultados formatados de cada análise
        if state.descriptive_results:
            formatted_response["analysis_results"]["descriptive"] = formatter.format_descriptive_analysis(state.descriptive_results)

        if state.pattern_results:
            formatted_response["analysis_results"]["pattern"] = formatter.format_pattern_analysis(state.pattern_results)

        if state.anomaly_results:
            formatted_response["analysis_results"]["anomaly"] = formatter.format_anomaly_analysis(state.anomaly_results)

        if state.relationship_results:
            formatted_response["analysis_results"]["relationship"] = formatter.format_relationship_analysis(state.relationship_results)

        # Adicionar código gerado se disponível
        if state.generated_code:
            formatted_response["generated_code"] = state.generated_code
            formatted_response["execution_results"] = state.execution_results

        state.response_data = formatted_response

        # Adicionar ao histórico de conversação
        state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "system_response",
            "content": state.final_insights,
            "has_visualizations": len(state.visualizations) > 0,
            "analyses_completed": list(state.completed_analyses)
        })

        log_data_operation(
            "response_formatted",
            {
                "response_sections": len(formatted_response["analysis_results"]),
                "has_visualizations": len(state.visualizations) > 0,
                "has_errors": state.has_errors()
            },
            "format_response"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"insights_preview": state.final_insights[:100] if state.final_insights else ""},
            "format_response_error"
        )
        state.add_error(f"Erro na formatação da resposta: {str(e)}")
        return state


def route_to_analysis_node(state: EDAState) -> EDAState:
    """
    Nó de roteamento - determina quais análises executar baseado na query.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com tipos de análise determinados
    """
    logger = get_logger("route_to_analysis")

    try:
        if state.has_errors():
            return state

        # Determinar tipos de análise necessários baseado na query
        query_lower = state.user_query.lower()
        required_analyses = set()

        # Palavras-chave para diferentes tipos de análise
        descriptive_keywords = ["describe", "summary", "overview", "basic", "estatisticas", "resumo"]
        pattern_keywords = ["pattern", "trend", "seasonal", "padrao", "tendencia"]
        anomaly_keywords = ["anomaly", "outlier", "unusual", "anomalia", "outliers"]
        relationship_keywords = ["correlation", "relationship", "correlacao", "relacao"]

        # Verificar palavras-chave na query
        if any(keyword in query_lower for keyword in descriptive_keywords):
            required_analyses.add("descriptive_analysis")

        if any(keyword in query_lower for keyword in pattern_keywords):
            required_analyses.add("pattern_analysis")

        if any(keyword in query_lower for keyword in anomaly_keywords):
            required_analyses.add("anomaly_analysis")

        if any(keyword in query_lower for keyword in relationship_keywords):
            required_analyses.add("relationship_analysis")

        # Se nenhuma análise específica foi identificada, fazer análise descritiva por padrão
        if not required_analyses:
            required_analyses.add("descriptive_analysis")

        # Atualizar estado com análises necessárias
        state.required_analyses = required_analyses

        log_data_operation(
            "analysis_routing_completed",
            {
                "required_analyses": list(required_analyses),
                "query_preview": state.user_query[:100]
            },
            "route_to_analysis"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"query_preview": state.user_query[:100]},
            "route_to_analysis_error"
        )
        state.add_error(f"Erro no roteamento de análise: {str(e)}")
        return state


def error_handler_node(state: EDAState) -> EDAState:
    """
    Nó de tratamento de erros - processa erros e prepara resposta de erro.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com tratamento de erro
    """
    logger = get_logger("error_handler")

    try:
        # Se não há erros, retorna o estado inalterado
        if not state.has_errors():
            return state

        # Log dos erros encontrados
        log_error_with_context(
            Exception("Multiple workflow errors"),
            {
                "error_count": len(state.errors),
                "errors": state.errors[:3],  # Log apenas os primeiros 3 erros
                "query_preview": state.user_query[:100] if state.user_query else ""
            },
            "workflow_handler"
        )

        # Preparar resposta de erro formatada
        error_response = {
            "success": False,
            "error_summary": "Ocorreram erros durante a análise dos dados.",
            "errors": state.errors,
            "timestamp": datetime.now().isoformat(),
            "query": state.user_query,
            "suggestions": [
                "Verifique se o arquivo CSV está bem formatado",
                "Tente uma consulta mais simples",
                "Certifique-se de que os dados são válidos"
            ]
        }

        # Se há alguns dados processados, incluir o que foi possível analisar
        if state.csv_data is not None:
            error_response["partial_data_info"] = {
                "shape": list(state.csv_data.shape),
                "columns": list(state.csv_data.columns),
                "data_types": state.csv_data.dtypes.astype(str).to_dict()
            }

        state.response_data = error_response
        state.final_insights = f"Análise interrompida devido a erros: {'; '.join(state.errors[:2])}"

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {"original_errors_count": len(state.errors) if state.errors else 0},
            "error_handler_failure"
        )

        # Erro crítico - criar resposta mínima
        state.response_data = {
            "success": False,
            "error_summary": "Erro crítico no sistema de análise",
            "timestamp": datetime.now().isoformat()
        }
        return state


# ====================================================================
# NOVOS NÓS: TOOL ORCHESTRATION (Sistema baseado em ferramentas)
# ====================================================================

def tool_orchestration_node(state: EDAState) -> EDAState:
    """
    Nó de orquestração de ferramentas - usa LLM para decidir quais ferramentas executar.

    Este é o nó central do novo sistema baseado em ferramentas.
    O LLM analisa a query e decide quais ferramentas estatísticas chamar.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resultados das ferramentas executadas
    """
    logger = get_logger("tool_orchestration")

    try:
        from ..tools import get_tool_registry
        from ..core.llm_manager import get_llm_manager

        if state.has_errors() or state.csv_data is None:
            return state

        logger.info("Iniciando orquestração de ferramentas")

        # Obter instâncias necessárias
        tool_registry = get_tool_registry()
        llm_manager = get_llm_manager()

        # Preparar informações do dataset para o LLM
        dataset_info = {
            'shape': state.csv_data.shape,
            'columns': list(state.csv_data.columns),
            'dtypes': {col: str(dtype) for col, dtype in state.csv_data.dtypes.items()},
            'total_rows': len(state.csv_data),
            'total_columns': len(state.csv_data.columns)
        }

        # Obter descrição de todas as ferramentas disponíveis
        available_tools = tool_registry.get_all_tools_description()

        logger.info(f"Ferramentas disponíveis: {len(available_tools)}")

        # LLM decide quais ferramentas usar
        tool_calls = llm_manager.get_tool_calls(
            user_query=state.user_query,
            dataset_info=dataset_info,
            available_tools=available_tools,
            context=None
        )

        logger.info(f"LLM sugeriu {len(tool_calls)} ferramentas")

        # Executar ferramentas
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get('tool_name')
            parameters = tool_call.get('parameters', {})

            logger.info(f"Executando ferramenta: {tool_name} com parâmetros: {parameters}")

            # Executar ferramenta
            result = tool_registry.execute_tool(
                tool_name=tool_name,
                df=state.csv_data,
                parameters=parameters,
                use_cache=True
            )

            tool_results.append({
                'tool_name': tool_name,
                'parameters': parameters,
                'result': result.to_dict()
            })

            logger.info(
                f"Ferramenta {tool_name} executada: "
                f"{'sucesso' if result.success else 'erro'} "
                f"(tempo: {result.execution_time:.3f}s)"
            )

        # Salvar resultados no estado
        state.tool_execution_results = tool_results

        log_data_operation(
            "tool_orchestration_completed",
            {
                "tools_executed": len(tool_results),
                "tools_list": [r['tool_name'] for r in tool_results],
                "total_execution_time": sum(r['result']['execution_time'] for r in tool_results)
            },
            "tool_orchestration"
        )

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {
                "query": state.user_query,
                "has_data": state.csv_data is not None
            },
            "tool_orchestration_error"
        )
        state.add_error(f"Erro na orquestração de ferramentas: {str(e)}")
        return state


def tool_synthesis_node(state: EDAState) -> EDAState:
    """
    Nó de síntese baseado em ferramentas - sintetiza resultados das ferramentas em resposta final.

    Este nó usa o LLM para interpretar os resultados das ferramentas e gerar
    uma resposta em linguagem natural para o usuário.

    Args:
        state: Estado atual do LangGraph

    Returns:
        Estado atualizado com resposta sintetizada
    """
    logger = get_logger("tool_synthesis")

    try:
        from ..core.llm_manager import get_llm_manager
        from ..core.memory_manager import get_memory_manager

        if state.has_errors():
            logger.warning("Síntese pulada: estado contém erros")
            return state

        llm_manager = get_llm_manager()
        memory_manager = get_memory_manager()

        # Recuperar contexto de memória se session_id existir
        memory_context = None
        if state.session_id:
            try:
                # Primeiro, verificar se a sessão existe
                session = memory_manager.get_session(state.session_id)
                if session:
                    memory_context = memory_manager.get_session_context(state.session_id)
                    if memory_context:
                        logger.info(f"Contexto de memória recuperado para sessão {state.session_id}: {memory_context.get('total_queries', 0)} queries anteriores")
                    else:
                        logger.info(f"Sessão {state.session_id} existe mas contexto está vazio")
                else:
                    logger.warning(f"Sessão {state.session_id} não encontrada, continuando sem contexto")
            except Exception as e:
                logger.warning(f"Erro ao recuperar contexto de memória: {e}, continuando sem contexto")

        # Verificar se há resultados de ferramentas
        if not state.tool_execution_results:
            # MODO CONCEITUAL: Nenhuma ferramenta foi executada
            # LLM responde usando apenas informações do schema
            logger.info("Modo conceitual ativado: nenhuma ferramenta executada")

            # Preparar informações do dataset
            dataset_info = {
                'shape': state.csv_data.shape if state.csv_data is not None else (0, 0),
                'columns': list(state.csv_data.columns) if state.csv_data is not None else [],
                'dtypes': {col: str(dtype) for col, dtype in state.csv_data.dtypes.items()} if state.csv_data is not None else {},
                'total_rows': len(state.csv_data) if state.csv_data is not None else 0,
                'total_columns': len(state.csv_data.columns) if state.csv_data is not None else 0
            }

            # LLM sintetiza resposta conceitual com contexto de memória
            synthesized_response = llm_manager.conceptual_synthesis(
                user_query=state.user_query,
                dataset_info=dataset_info,
                context=memory_context  # Passar contexto de memória
            )

            log_data_operation(
                "conceptual_synthesis_completed",
                {
                    "response_length": len(synthesized_response),
                    "mode": "conceptual"
                },
                "tool_synthesis"
            )

        else:
            # MODO FERRAMENTAS: Ferramentas foram executadas
            logger.info(f"Modo ferramentas ativado: {len(state.tool_execution_results)} ferramentas executadas")

            # LLM sintetiza resultados em linguagem natural com contexto de memória
            synthesized_response = llm_manager.synthesize_tool_results(
                user_query=state.user_query,
                tool_results=state.tool_execution_results,
                context=memory_context  # Passar contexto de memória
            )

            log_data_operation(
                "tool_synthesis_completed",
                {
                    "response_length": len(synthesized_response),
                    "tools_synthesized": len(state.tool_execution_results),
                    "mode": "tools"
                },
                "tool_synthesis"
            )

        # Detectar visualizações nos resultados das ferramentas
        visualizations = []
        for tool_result in state.tool_execution_results:
            result_data = tool_result.get('result', {}).get('data', {})

            # Verificar se resultado contém visualização (image_base64)
            if 'image_base64' in result_data:
                visualization = {
                    'type': result_data.get('plot_type', 'unknown'),
                    'title': result_data.get('title', 'Visualização'),
                    'image_base64': result_data['image_base64'],
                    'metadata': {
                        'column': result_data.get('column'),
                        'stats': result_data.get('stats'),
                        'outliers_info': result_data.get('outliers_info'),
                        'correlation_matrix': result_data.get('correlation_matrix')
                    }
                }
                visualizations.append(visualization)
                logger.info(f"Visualização detectada: {result_data.get('plot_type')} para {result_data.get('column')}")

        # Salvar visualizações no estado
        state.visualizations = visualizations

        # Salvar resposta sintetizada
        state.tool_based_response = synthesized_response
        state.final_insights = synthesized_response  # Compatibilidade com sistema antigo

        # Salvar insights na sessão de memória se session_id existir
        if state.session_id and synthesized_response:
            try:
                # Verificar se a sessão existe antes de tentar salvar
                session = memory_manager.get_session(state.session_id)
                if session:
                    from ..models.graph_schema import DataInsight
                    from ..models.enums import EDAAnalysisType

                    # Criar insight básico com a resposta sintetizada
                    insight = DataInsight(
                        insight_text=synthesized_response[:500],  # Limitar tamanho
                        confidence=0.8,  # Confiança alta para resultados de ferramentas
                        analysis_type=EDAAnalysisType.DESCRIPTIVE,
                        importance="high"
                    )

                    memory_manager.save_insights(
                        session_id=state.session_id,
                        insights=[insight]
                    )
                    logger.info(f"Insights salvos na sessão de memória: {state.session_id}")
                else:
                    logger.warning(f"Sessão {state.session_id} não encontrada, não foi possível salvar insights")
            except Exception as e:
                logger.warning(f"Erro ao salvar insights na memória: {e}")

        logger.info(f"Síntese completada com sucesso ({len(visualizations)} visualizações detectadas)")

        return state

    except Exception as e:
        log_error_with_context(
            e,
            {
                "query": state.user_query,
                "tool_results_count": len(state.tool_execution_results) if state.tool_execution_results else 0
            },
            "tool_synthesis_error"
        )
        state.add_error(f"Erro na síntese de resultados: {str(e)}")
        return state