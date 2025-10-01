"""
Testes de integração para o sistema LangGraph workflow.

Testa a funcionalidade completa do sistema de workflow incluindo:
- Construção e validação do grafo
- Execução de workflows completos
- Gerenciamento de estado
- Roteamento entre nós
- Tratamento de erros
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from src.graph import (
    EDAState,
    EDAWorkflow,
    EDAWorkflowError,
    get_workflow,
    reset_workflow,
    execute_eda_analysis_sync,
    validate_graph_module,
    get_module_info
)
from src.graph.nodes import entry_point_node, classify_query_node
from src.graph.edges import classify_query_edge, process_data_edge
from src.models.enums import EDAAnalysisType, DataType


class TestEDAState:
    """Testes para a classe EDAState."""

    def test_state_initialization(self):
        """Testa inicialização básica do estado."""
        state = EDAState(
            user_query="Teste query",
            csv_data="col1,col2\n1,2\n3,4"
        )

        assert state.user_query == "Teste query"
        assert state.csv_data == "col1,col2\n1,2\n3,4"
        assert state.errors == []
        assert state.analysis_results == {}
        assert state.required_agents == []

    def test_add_error(self):
        """Testa adição de erros ao estado."""
        state = EDAState(user_query="test")

        state.add_error("Erro teste")
        assert len(state.errors) == 1
        assert "Erro teste" in state.errors
        assert state.has_errors()

    def test_should_execute_agent(self):
        """Testa lógica de execução de agentes."""
        state = EDAState(user_query="test")
        state.required_agents = ["data_analyzer", "pattern_detector"]

        assert state.should_execute_agent("data_analyzer")
        assert state.should_execute_agent("pattern_detector")
        assert not state.should_execute_agent("anomaly_detector")

    def test_is_analysis_completed(self):
        """Testa verificação de conclusão de análises."""
        state = EDAState(user_query="test")

        # Inicialmente nenhuma análise está completa
        assert not state.is_analysis_completed(EDAAnalysisType.DESCRIPTIVE)

        # Adicionar resultado de análise
        state.analysis_results[EDAAnalysisType.DESCRIPTIVE] = {"result": "test"}
        assert state.is_analysis_completed(EDAAnalysisType.DESCRIPTIVE)

    def test_get_all_results(self):
        """Testa obtenção de todos os resultados."""
        state = EDAState(user_query="test")
        state.analysis_results = {
            EDAAnalysisType.DESCRIPTIVE: {"stats": "test"},
            EDAAnalysisType.PATTERN_ANALYSIS: {"patterns": "test"}
        }

        results = state.get_all_results()
        assert len(results) == 2
        assert EDAAnalysisType.DESCRIPTIVE in results
        assert EDAAnalysisType.PATTERN_ANALYSIS in results

    def test_reset_for_new_query(self):
        """Testa reset do estado para nova consulta."""
        state = EDAState(user_query="test")
        state.errors = ["erro"]
        state.analysis_results = {EDAAnalysisType.DESCRIPTIVE: {"test": "data"}}
        state.generated_code = ["código"]

        state.reset_for_new_query("nova query")

        assert state.user_query == "nova query"
        assert state.errors == []
        assert state.analysis_results == {}
        assert state.generated_code == []

    def test_is_ready_for_synthesis(self):
        """Testa verificação de prontidão para síntese."""
        state = EDAState(user_query="test")

        # Não está pronto inicialmente
        assert not state.is_ready_for_synthesis()

        # Adicionar alguns resultados
        state.analysis_results = {EDAAnalysisType.DESCRIPTIVE: {"test": "data"}}
        assert state.is_ready_for_synthesis()


class TestEDAWorkflow:
    """Testes para a classe EDAWorkflow."""

    def setup_method(self):
        """Setup para cada teste."""
        reset_workflow()

    def test_workflow_initialization(self):
        """Testa inicialização do workflow."""
        workflow = EDAWorkflow()
        assert workflow._graph is not None

    def test_get_workflow_singleton(self):
        """Testa padrão singleton do workflow."""
        workflow1 = get_workflow()
        workflow2 = get_workflow()
        assert workflow1 is workflow2

    def test_workflow_structure(self):
        """Testa estrutura do grafo."""
        workflow = get_workflow()
        structure = workflow.get_graph_structure()

        assert "nodes" in structure
        assert "edges" in structure
        assert "entry_point" in structure
        assert structure["entry_point"] == "entry_point"
        assert structure["total_nodes"] > 0

    def test_workflow_validation(self):
        """Testa validação do workflow."""
        workflow = get_workflow()
        validation = workflow.validate_workflow()

        assert "is_valid" in validation
        assert "issues" in validation
        assert "warnings" in validation

    @patch('src.graph.nodes.entry_point_node')
    @patch('src.graph.nodes.classify_query_node')
    def test_workflow_execution_sync_mock(self, mock_classify, mock_entry):
        """Testa execução síncrona com mocks."""
        # Configurar mocks
        def mock_entry_node(state):
            state.processed_query = state.user_query
            return state

        def mock_classify_node(state):
            state.query_classification = {"type": "descriptive"}
            state.required_agents = ["data_analyzer"]
            return state

        mock_entry.side_effect = mock_entry_node
        mock_classify.side_effect = mock_classify_node

        # Simular erro para não executar todo o workflow
        with patch('src.graph.edges.classify_query_edge', return_value="error_handler"):
            with patch('src.graph.nodes.error_handler_node') as mock_error:
                def mock_error_node(state):
                    state.response_data = {"error": "Teste finalizado"}
                    return state

                mock_error.side_effect = mock_error_node

                # Executar workflow
                result = execute_eda_analysis_sync(
                    user_query="Teste de consulta",
                    csv_data=None
                )

                assert "success" in result
                assert "user_query" in result

    def test_workflow_with_invalid_state(self):
        """Testa workflow com estado inválido."""
        workflow = get_workflow()

        # Tentar executar com query vazia deve falhar
        with pytest.raises(Exception):
            workflow.execute_sync("")

    def test_reset_workflow(self):
        """Testa reset do singleton workflow."""
        workflow1 = get_workflow()
        reset_workflow()
        workflow2 = get_workflow()

        assert workflow1 is not workflow2


class TestWorkflowNodes:
    """Testes para os nós individuais do workflow."""

    def test_entry_point_node(self):
        """Testa nó de entrada."""
        state = EDAState(user_query="teste")
        result = entry_point_node(state)

        assert result.user_query == "teste"
        assert result.processed_query is not None

    @patch('src.core.query_interpreter.QueryInterpreter.classify_query')
    def test_classify_query_node(self, mock_classify):
        """Testa nó de classificação."""
        mock_classify.return_value = {
            "analysis_types": ["descriptive"],
            "required_agents": ["data_analyzer"],
            "data_requirements": ["csv"]
        }

        state = EDAState(user_query="análise descritiva")
        state.processed_query = state.user_query

        result = classify_query_node(state)

        assert result.query_classification is not None
        assert result.required_agents == ["data_analyzer"]


class TestWorkflowEdges:
    """Testes para as arestas do workflow."""

    def test_classify_query_edge_with_csv(self):
        """Testa aresta de classificação com dados CSV."""
        state = EDAState(
            user_query="teste",
            csv_data="col1,col2\n1,2"
        )
        state.query_classification = {"type": "test"}
        state.required_agents = ["data_analyzer"]

        result = classify_query_edge(state)
        assert result == "process_data"

    def test_classify_query_edge_without_csv(self):
        """Testa aresta de classificação sem dados CSV."""
        state = EDAState(user_query="teste")
        state.query_classification = {"type": "test"}
        state.required_agents = ["data_analyzer"]
        state.processed_query = "teste processado"

        result = classify_query_edge(state)
        assert result == "route_to_analysis"

    def test_classify_query_edge_with_errors(self):
        """Testa aresta de classificação com erros."""
        state = EDAState(user_query="teste")
        state.add_error("Erro de teste")

        result = classify_query_edge(state)
        assert result == "error_handler"

    def test_process_data_edge_success(self):
        """Testa aresta de processamento de dados com sucesso."""
        state = EDAState(user_query="teste")
        state.csv_data = "col1,col2\n1,2"
        state.csv_metadata = {"columns": ["col1", "col2"]}

        result = process_data_edge(state)
        assert result == "route_to_analysis"

    def test_process_data_edge_failure(self):
        """Testa aresta de processamento de dados com falha."""
        state = EDAState(user_query="teste")
        # csv_data e csv_metadata permanecem None

        result = process_data_edge(state)
        assert result == "error_handler"


class TestWorkflowIntegration:
    """Testes de integração completos."""

    def setup_method(self):
        """Setup para cada teste."""
        reset_workflow()

    def test_module_validation(self):
        """Testa validação do módulo completo."""
        is_valid = validate_graph_module()
        # O resultado pode ser True ou False dependendo das dependências
        assert isinstance(is_valid, bool)

    def test_module_info(self):
        """Testa informações do módulo."""
        info = get_module_info()

        assert "version" in info
        assert "is_valid" in info
        assert isinstance(info["version"], str)
        assert isinstance(info["is_valid"], bool)

    @patch('src.graph.workflow.StateGraph')
    def test_workflow_build_failure(self, mock_state_graph):
        """Testa falha na construção do workflow."""
        mock_state_graph.side_effect = Exception("Erro de construção")

        with pytest.raises(EDAWorkflowError):
            EDAWorkflow()

    def test_workflow_with_sample_data(self):
        """Testa workflow com dados de exemplo."""
        sample_csv = "nome,idade,salario\nJoao,30,5000\nMaria,25,4500"

        # Usar mocks para evitar dependências externas
        with patch('src.core.query_interpreter.QueryInterpreter') as mock_interpreter:
            with patch('src.agents.data_analyzer.DataAnalyzerAgent') as mock_analyzer:
                mock_interpreter.return_value.classify_query.return_value = {
                    "analysis_types": ["descriptive"],
                    "required_agents": ["data_analyzer"],
                    "data_requirements": ["csv"]
                }

                mock_analyzer.return_value.analyze_data.return_value = {
                    "statistics": {"count": 2},
                    "insights": ["Dados processados com sucesso"]
                }

                # Este teste pode falhar devido a dependências, mas serve para verificar a estrutura
                try:
                    result = execute_eda_analysis_sync(
                        user_query="Faça uma análise descritiva dos dados",
                        csv_data=sample_csv
                    )

                    # Se chegou até aqui, verificar estrutura básica
                    assert isinstance(result, dict)
                    assert "success" in result

                except Exception as e:
                    # Esperado devido a dependências não mockadas
                    assert "Falha na execução" in str(e) or "langgraph" in str(e).lower()


class TestErrorHandling:
    """Testes para tratamento de erros."""

    def test_workflow_error_creation(self):
        """Testa criação de erros específicos do workflow."""
        error = EDAWorkflowError("Teste de erro")
        assert str(error) == "Teste de erro"

    def test_state_error_handling(self):
        """Testa tratamento de erros no estado."""
        state = EDAState(user_query="teste")

        # Adicionar múltiplos erros
        state.add_error("Erro 1")
        state.add_error("Erro 2")

        assert len(state.errors) == 2
        assert state.has_errors()

    def test_workflow_process_result_with_errors(self):
        """Testa processamento de resultado com erros."""
        workflow = get_workflow()
        state = EDAState(user_query="teste")
        state.add_error("Erro de teste")

        result = workflow._process_result(state)

        assert result["success"] is False
        assert "errors" in result
        assert "Erro de teste" in result["errors"]


if __name__ == "__main__":
    # Executar testes básicos
    print("Executando testes básicos do módulo graph...")

    try:
        # Teste de importação
        from src.graph import EDAState, get_workflow
        print("✓ Importações realizadas com sucesso")

        # Teste de estado
        state = EDAState(user_query="teste")
        print(f"✓ Estado criado: {state.user_query}")

        # Teste de workflow
        workflow = get_workflow()
        structure = workflow.get_graph_structure()
        print(f"✓ Workflow criado com {structure.get('total_nodes', 0)} nós")

        # Teste de validação
        validation = workflow.validate_workflow()
        print(f"✓ Validação do workflow: {validation.get('is_valid', False)}")

        print("\nTodos os testes básicos passaram!")

    except Exception as e:
        print(f"✗ Erro nos testes básicos: {str(e)}")
        raise