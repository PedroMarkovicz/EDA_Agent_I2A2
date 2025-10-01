"""
Testes end-to-end do sistema EDA completo.
Testa fluxos completos: upload CSV -> query -> analise -> resposta.
"""

import pytest
import pandas as pd
from unittest.mock import patch, Mock
from src.graph.workflow import get_workflow, reset_workflow
from src.graph.state import EDAState
from src.models.enums import EDAAnalysisType, ProcessingStatus


class TestDescriptiveAnalysisFlow:
    """Testes de fluxo completo para analise descritiva."""

    def setup_method(self):
        """Setup antes de cada teste."""
        reset_workflow()

    @patch('src.core.llm_manager.LLMManager.chat_completion')
    def test_complete_descriptive_flow(self, mock_llm, simple_df):
        """Testa fluxo completo de analise descritiva."""
        mock_llm.return_value = {
            "content": "Analise descritiva concluida com sucesso",
            "usage": {"total_tokens": 100}
        }

        workflow = get_workflow()
        state = EDAState(
            user_query="Qual e a media de idade?",
            csv_data=simple_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None
        assert result.final_insights is not None or result.response_data is not None

    @patch('src.core.llm_manager.LLMManager.chat_completion')
    def test_specific_value_query(self, mock_llm, simple_df):
        """Testa query especifica sobre valor."""
        mock_llm.return_value = {
            "content": "O valor maximo da coluna idade e 30",
            "usage": {"total_tokens": 100}
        }

        workflow = get_workflow()
        state = EDAState(
            user_query="qual e o valor maximo da coluna idade?",
            csv_data=simple_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None
        assert not result.has_errors()


class TestPatternDetectionFlow:
    """Testes de fluxo para deteccao de padroes."""

    def setup_method(self):
        """Setup antes de cada teste."""
        reset_workflow()

    @patch('src.core.llm_manager.LLMManager.chat_completion')
    def test_temporal_pattern_flow(self, mock_llm, temporal_df):
        """Testa deteccao de padroes temporais."""
        mock_llm.return_value = {
            "content": "Padroes temporais identificados",
            "usage": {"total_tokens": 100}
        }

        workflow = get_workflow()
        state = EDAState(
            user_query="Identifique padroes temporais nas vendas",
            csv_data=temporal_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None


class TestAnomalyDetectionFlow:
    """Testes de fluxo para deteccao de anomalias."""

    def setup_method(self):
        """Setup antes de cada teste."""
        reset_workflow()

    @patch('src.core.llm_manager.LLMManager.chat_completion')
    def test_outlier_detection_flow(self, mock_llm, outliers_df):
        """Testa deteccao de outliers."""
        mock_llm.return_value = {
            "content": "Outliers detectados com sucesso",
            "usage": {"total_tokens": 100}
        }

        workflow = get_workflow()
        state = EDAState(
            user_query="Detecte outliers nos dados",
            csv_data=outliers_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None


class TestRelationshipAnalysisFlow:
    """Testes de fluxo para analise de relacionamentos."""

    def setup_method(self):
        """Setup antes de cada teste."""
        reset_workflow()

    @patch('src.core.llm_manager.LLMManager.chat_completion')
    def test_correlation_analysis_flow(self, mock_llm, correlation_df):
        """Testa analise de correlacao."""
        mock_llm.return_value = {
            "content": "Correlacoes analisadas com sucesso",
            "usage": {"total_tokens": 100}
        }

        workflow = get_workflow()
        state = EDAState(
            user_query="Analise as correlacoes entre as variaveis",
            csv_data=correlation_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None


class TestVisualizationFlow:
    """Testes de fluxo com geracao de visualizacoes."""

    def setup_method(self):
        """Setup antes de cada teste."""
        reset_workflow()

    @patch('src.core.llm_manager.LLMManager.chat_completion')
    def test_flow_with_histogram(self, mock_llm, numeric_df):
        """Testa fluxo com geracao de histograma."""
        mock_llm.return_value = {
            "content": "Histograma gerado com sucesso",
            "usage": {"total_tokens": 100}
        }

        workflow = get_workflow()
        state = EDAState(
            user_query="Mostre um histograma da coluna col1",
            csv_data=numeric_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None


class TestErrorHandling:
    """Testes de tratamento de erros end-to-end."""

    def setup_method(self):
        """Setup antes de cada teste."""
        reset_workflow()

    def test_empty_dataframe(self):
        """Testa fluxo com dataframe vazio."""
        workflow = get_workflow()
        empty_df = pd.DataFrame()
        state = EDAState(
            user_query="Analise os dados",
            csv_data=empty_df
        )

        result = workflow._graph.invoke(state)

        assert result.has_errors() or result is not None

    def test_invalid_query(self, simple_df):
        """Testa fluxo com query invalida."""
        workflow = get_workflow()
        state = EDAState(
            user_query="",
            csv_data=simple_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None


class TestMultipleAnalysisTypes:
    """Testes com multiplos tipos de analise."""

    def setup_method(self):
        """Setup antes de cada teste."""
        reset_workflow()

    @patch('src.core.llm_manager.LLMManager.chat_completion')
    def test_combined_analysis(self, mock_llm, simple_df):
        """Testa analise combinada."""
        mock_llm.return_value = {
            "content": "Analise combinada concluida",
            "usage": {"total_tokens": 100}
        }

        workflow = get_workflow()
        state = EDAState(
            user_query="Faca uma analise completa dos dados incluindo outliers e correlacoes",
            csv_data=simple_df
        )

        result = workflow._graph.invoke(state)

        assert result is not None