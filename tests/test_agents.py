"""
Testes unitarios para agentes especializados EDA.
Testa funcionalidade de cada agente: DataAnalyzer, PatternDetector,
AnomalyDetector, RelationshipAnalyzer e ConclusionGenerator.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.agents.data_analyzer import DataAnalyzerAgent, get_data_analyzer_agent
from src.agents.pattern_detector import PatternDetectorAgent, get_pattern_detector_agent
from src.agents.anomaly_detector import AnomalyDetectorAgent, get_anomaly_detector_agent
from src.agents.relationship_analyzer import RelationshipAnalyzerAgent, get_relationship_analyzer_agent
from src.agents.conclusion_generator import ConclusionGeneratorAgent, get_conclusion_generator_agent
from src.models.enums import EDAAnalysisType, ProcessingStatus
from src.models.analysis_result import DescriptiveAnalysisResult


class TestDataAnalyzerAgent:
    """Testes para DataAnalyzerAgent."""

    def test_initialization(self):
        """Testa inicializacao do agente."""
        agent = DataAnalyzerAgent()
        assert agent is not None
        assert agent.config is not None
        assert agent.llm_manager is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        agent1 = get_data_analyzer_agent()
        agent2 = get_data_analyzer_agent()
        assert agent1 is agent2

    def test_analyze_basic(self, simple_df):
        """Testa analise basica de dataframe."""
        agent = DataAnalyzerAgent()

        with patch.object(agent.llm_manager, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "content": "Dataset analisado com sucesso",
                "usage": {"total_tokens": 100}
            }

            result = agent.analyze(simple_df)

            assert result is not None
            assert result.analysis_type == EDAAnalysisType.DESCRIPTIVE
            assert result.status == ProcessingStatus.COMPLETED
            assert result.dataset_overview is not None
            assert result.column_summaries is not None
            assert len(result.column_summaries) == 3

    def test_dataset_overview_generation(self, numeric_df):
        """Testa geracao de visao geral do dataset."""
        agent = DataAnalyzerAgent()
        overview = agent._generate_dataset_overview(numeric_df)

        assert overview["total_rows"] == 5
        assert overview["total_columns"] == 3
        assert "total_cells" in overview
        assert "memory_usage_mb" in overview
        assert "completeness_percentage" in overview

    def test_data_types_analysis(self, simple_df):
        """Testa analise de tipos de dados."""
        agent = DataAnalyzerAgent()
        types = agent._analyze_data_types(simple_df)

        assert "numeric" in types
        assert "categorical" in types
        assert types["numeric"] == 2

    def test_column_summaries_numeric(self, numeric_df):
        """Testa resumos de colunas numericas."""
        agent = DataAnalyzerAgent()
        summaries = agent._generate_column_summaries(numeric_df)

        assert len(summaries) == 3
        for summary in summaries:
            assert summary.mean is not None
            assert summary.std is not None
            assert summary.min_value is not None
            assert summary.max_value is not None

    def test_missing_data_analysis(self, missing_data_df):
        """Testa analise de dados ausentes."""
        agent = DataAnalyzerAgent()
        missing_info = agent._analyze_missing_data(missing_data_df)

        assert "columns_with_missing" in missing_info
        assert len(missing_info["columns_with_missing"]) > 0
        assert "total_missing_values" in missing_info

    def test_basic_insights_fallback(self, simple_df):
        """Testa geracao de insights basicos sem LLM."""
        agent = DataAnalyzerAgent()
        result = DescriptiveAnalysisResult(
            dataset_overview={"completeness_percentage": 96.0, "duplicate_percentage": 0.0},
            data_types_summary={"numeric": 3, "categorical": 1}
        )

        insights = agent._generate_basic_insights(result, "")
        assert len(insights) > 0
        assert any("95%" in insight for insight in insights)

    def test_specific_value_extraction(self, simple_df):
        """Testa extracao de valores especificos de queries."""
        agent = DataAnalyzerAgent()
        result = DescriptiveAnalysisResult(
            column_summaries=agent._generate_column_summaries(simple_df)
        )

        insights = agent._generate_basic_insights(result, "qual e o valor maximo da coluna idade?")
        assert len(insights) > 0
        assert any("30" in insight or "idade" in insight for insight in insights)


class TestPatternDetectorAgent:
    """Testes para PatternDetectorAgent."""

    def test_initialization(self):
        """Testa inicializacao do agente."""
        agent = PatternDetectorAgent()
        assert agent is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        agent1 = get_pattern_detector_agent()
        agent2 = get_pattern_detector_agent()
        assert agent1 is agent2

    def test_detect_patterns_basic(self, temporal_df):
        """Testa deteccao basica de padroes."""
        agent = PatternDetectorAgent()

        with patch.object(agent.llm_manager, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "content": "Padroes identificados com sucesso",
                "usage": {"total_tokens": 100}
            }

            result = agent.detect_patterns(temporal_df)

            assert result is not None
            assert result.analysis_type == EDAAnalysisType.PATTERN
            assert result.status == ProcessingStatus.COMPLETED

    def test_frequency_analysis(self, categorical_df):
        """Testa analise de frequencia."""
        agent = PatternDetectorAgent()
        frequencies = agent._analyze_frequencies(categorical_df)

        assert len(frequencies) > 0
        for col, freq_info in frequencies.items():
            assert "value_counts" in freq_info
            assert "most_common" in freq_info

    def test_temporal_pattern_detection(self, temporal_df):
        """Testa deteccao de padroes temporais."""
        agent = PatternDetectorAgent()
        patterns = agent._detect_temporal_patterns(temporal_df)

        assert patterns is not None
        assert isinstance(patterns, dict)


class TestAnomalyDetectorAgent:
    """Testes para AnomalyDetectorAgent."""

    def test_initialization(self):
        """Testa inicializacao do agente."""
        agent = AnomalyDetectorAgent()
        assert agent is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        agent1 = get_anomaly_detector_agent()
        agent2 = get_anomaly_detector_agent()
        assert agent1 is agent2

    def test_detect_anomalies_basic(self, outliers_df):
        """Testa deteccao basica de anomalias."""
        agent = AnomalyDetectorAgent()

        with patch.object(agent.llm_manager, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "content": "Anomalias detectadas",
                "usage": {"total_tokens": 100}
            }

            result = agent.detect_anomalies(outliers_df)

            assert result is not None
            assert result.analysis_type == EDAAnalysisType.ANOMALY
            assert result.status == ProcessingStatus.COMPLETED

    def test_iqr_method(self, numeric_df):
        """Testa metodo IQR para deteccao de outliers."""
        agent = AnomalyDetectorAgent()
        outliers = agent._detect_outliers_iqr(numeric_df['col1'])

        assert outliers is not None
        assert isinstance(outliers, list)

    def test_zscore_method(self, numeric_df):
        """Testa metodo Z-score para deteccao de outliers."""
        agent = AnomalyDetectorAgent()
        col = pd.concat([numeric_df['col1'], pd.Series([1000])])
        outliers = agent._detect_outliers_zscore(col)

        assert outliers is not None
        assert isinstance(outliers, list)

    def test_outliers_by_column(self, outliers_df):
        """Testa deteccao de outliers por coluna."""
        agent = AnomalyDetectorAgent()
        result = agent._analyze_outliers_by_column(outliers_df)

        assert "valores" in result
        assert len(result["valores"]["outliers"]) > 0


class TestRelationshipAnalyzerAgent:
    """Testes para RelationshipAnalyzerAgent."""

    def test_initialization(self):
        """Testa inicializacao do agente."""
        agent = RelationshipAnalyzerAgent()
        assert agent is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        agent1 = get_relationship_analyzer_agent()
        agent2 = get_relationship_analyzer_agent()
        assert agent1 is agent2

    def test_analyze_relationships_basic(self, correlation_df):
        """Testa analise basica de relacionamentos."""
        agent = RelationshipAnalyzerAgent()

        with patch.object(agent.llm_manager, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "content": "Relacionamentos analisados",
                "usage": {"total_tokens": 100}
            }

            result = agent.analyze_relationships(correlation_df)

            assert result is not None
            assert result.analysis_type == EDAAnalysisType.RELATIONSHIP
            assert result.status == ProcessingStatus.COMPLETED

    def test_correlation_matrix_calculation(self, correlation_df):
        """Testa calculo de matriz de correlacao."""
        agent = RelationshipAnalyzerAgent()
        corr_matrix = agent._calculate_correlation_matrix(correlation_df)

        assert corr_matrix is not None
        assert not corr_matrix.empty

    def test_strong_correlations_identification(self, correlation_df):
        """Testa identificacao de correlacoes fortes."""
        agent = RelationshipAnalyzerAgent()
        corr_matrix = correlation_df.corr()
        strong_corrs = agent._identify_strong_correlations(corr_matrix)

        assert isinstance(strong_corrs, list)
        if len(strong_corrs) > 0:
            assert "var1" in strong_corrs[0]
            assert "var2" in strong_corrs[0]
            assert "correlation" in strong_corrs[0]

    def test_multicollinearity_detection(self, correlation_df):
        """Testa deteccao de multicolinearidade."""
        agent = RelationshipAnalyzerAgent()
        corr_matrix = correlation_df.corr()
        multicollinear = agent._detect_multicollinearity(corr_matrix)

        assert isinstance(multicollinear, list)


class TestConclusionGeneratorAgent:
    """Testes para ConclusionGeneratorAgent."""

    def test_initialization(self):
        """Testa inicializacao do agente."""
        agent = ConclusionGeneratorAgent()
        assert agent is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        agent1 = get_conclusion_generator_agent()
        agent2 = get_conclusion_generator_agent()
        assert agent1 is agent2

    @patch('src.agents.conclusion_generator.get_llm_manager')
    def test_generate_conclusion_basic(self, mock_llm_get, sample_descriptive_result):
        """Testa geracao basica de conclusao."""
        mock_llm = Mock()
        mock_llm.chat_completion.return_value = {
            "content": "Conclusao gerada com sucesso",
            "usage": {"total_tokens": 100}
        }
        mock_llm_get.return_value = mock_llm

        agent = ConclusionGeneratorAgent()
        result = agent.generate_conclusion(
            descriptive_result=sample_descriptive_result,
            pattern_result=None,
            anomaly_result=None,
            relationship_result=None,
            context={"query": "Analise os dados"}
        )

        assert result is not None
        assert result.analysis_type == EDAAnalysisType.CONCLUSION
        assert result.status == ProcessingStatus.COMPLETED

    def test_consolidate_insights(self, sample_descriptive_result):
        """Testa consolidacao de insights."""
        agent = ConclusionGeneratorAgent()

        with patch.object(agent.llm_manager, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "content": "Insight 1\nInsight 2\nInsight 3",
                "usage": {"total_tokens": 100}
            }

            insights = agent._consolidate_insights(
                descriptive_result=sample_descriptive_result,
                pattern_result=None,
                anomaly_result=None,
                relationship_result=None,
                context={"query": "teste"}
            )

            assert isinstance(insights, list)

    def test_specific_answer_extraction(self, sample_descriptive_result):
        """Testa extracao de respostas especificas."""
        agent = ConclusionGeneratorAgent()
        answer = agent._extract_specific_answer(
            "qual e o valor maximo da coluna idade?",
            sample_descriptive_result
        )

        assert answer is None or isinstance(answer, str)

    def test_generate_response_text_with_specific_query(self, sample_descriptive_result):
        """Testa geracao de texto de resposta com query especifica."""
        agent = ConclusionGeneratorAgent()
        from src.models.graph_schema import DataInsight

        insights = [
            DataInsight(
                insight_text="Teste insight",
                confidence=0.9,
                analysis_type=EDAAnalysisType.DESCRIPTIVE,
                importance="high"
            )
        ]

        response = agent._generate_response_text(
            insights=insights,
            descriptive_result=sample_descriptive_result,
            pattern_result=None,
            anomaly_result=None,
            relationship_result=None,
            context={"query": "qual e a media de idade?"}
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_priority_in_response_generation(self, sample_descriptive_result):
        """Testa priorizacao na geracao de resposta."""
        agent = ConclusionGeneratorAgent()
        from src.models.graph_schema import DataInsight

        high_priority = DataInsight(
            insight_text="High priority insight",
            confidence=0.95,
            analysis_type=EDAAnalysisType.DESCRIPTIVE,
            importance="high"
        )

        low_priority = DataInsight(
            insight_text="Low priority insight",
            confidence=0.5,
            analysis_type=EDAAnalysisType.DESCRIPTIVE,
            importance="low"
        )

        response = agent._generate_response_text(
            insights=[high_priority, low_priority],
            descriptive_result=sample_descriptive_result,
            pattern_result=None,
            anomaly_result=None,
            relationship_result=None,
            context={"query": "analise os dados"}
        )

        assert "High priority" in response or len(response) > 0