"""
Testes unitários para os modelos de dados do sistema EDA.
Valida se todos os modelos podem ser criados e serializados corretamente.
"""

import pytest
import pandas as pd
from datetime import datetime

from src.models.enums import (
    EDAAnalysisType,
    QueryIntentType,
    ProcessingStatus,
    VisualizationType,
)
from src.models.query_schema import UserQuery, QueryEntity, QueryClassification
from src.models.analysis_result import (
    StatisticalSummary,
    VisualizationResult,
    DescriptiveAnalysisResult,
    PatternAnalysisResult,
    AnomalyAnalysisResult,
    RelationshipAnalysisResult,
)
from src.models.graph_schema import (
    DataInsight,
    ExecutiveSummary,
    ConsolidatedResults,
    MemoryContext,
    FinalResponse,
)
from src.graph.state import EDAState


class TestEnums:
    """Testes para as enumerações do sistema."""

    def test_eda_analysis_type(self):
        """Testa enumeração EDAAnalysisType."""
        assert EDAAnalysisType.DESCRIPTIVE == "descriptive"
        assert EDAAnalysisType.PATTERN == "pattern"
        assert EDAAnalysisType.ANOMALY == "anomaly"
        assert EDAAnalysisType.RELATIONSHIP == "relationship"
        assert EDAAnalysisType.CONCLUSION == "conclusion"

    def test_query_intent_type(self):
        """Testa enumeração QueryIntentType."""
        assert QueryIntentType.DATA_OVERVIEW == "data_overview"
        assert QueryIntentType.STATISTICAL_SUMMARY == "statistical_summary"
        assert QueryIntentType.CORRELATION_ANALYSIS == "correlation_analysis"

    def test_processing_status(self):
        """Testa enumeração ProcessingStatus."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.IN_PROGRESS == "in_progress"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"

    def test_visualization_type(self):
        """Testa enumeração VisualizationType."""
        assert VisualizationType.HISTOGRAM == "histogram"
        assert VisualizationType.BOXPLOT == "boxplot"
        assert VisualizationType.SCATTER_PLOT == "scatter_plot"
        assert VisualizationType.CORRELATION_HEATMAP == "correlation_heatmap"


class TestQueryModels:
    """Testes para modelos de consulta."""

    def test_query_entity_creation(self):
        """Testa criação de QueryEntity."""
        entity = QueryEntity(
            name="age",
            entity_type="column",
            value="25"
        )
        assert entity.name == "age"
        assert entity.entity_type == "column"
        assert entity.value == "25"

    def test_user_query_creation(self):
        """Testa criação de UserQuery."""
        query = UserQuery(
            query_text="Show me the distribution of age",
            intent_type=QueryIntentType.DISTRIBUTION_ANALYSIS,
            analysis_types=[EDAAnalysisType.DESCRIPTIVE],
            requires_visualization=True,
            visualization_type=VisualizationType.HISTOGRAM
        )
        assert query.query_text == "Show me the distribution of age"
        assert query.intent_type == QueryIntentType.DISTRIBUTION_ANALYSIS
        assert EDAAnalysisType.DESCRIPTIVE in query.analysis_types
        assert query.requires_visualization is True
        assert query.visualization_type == VisualizationType.HISTOGRAM

    def test_query_classification(self):
        """Testa QueryClassification."""
        query = UserQuery(query_text="Test query")
        classification = QueryClassification(
            query=query,
            confidence_score=0.95,
            required_agents=[EDAAnalysisType.DESCRIPTIVE],
            processing_order=[EDAAnalysisType.DESCRIPTIVE],
            estimated_complexity="medium"
        )
        assert classification.confidence_score == 0.95
        assert classification.estimated_complexity == "medium"
        assert EDAAnalysisType.DESCRIPTIVE in classification.required_agents

    def test_user_query_defaults(self):
        """Testa valores padrão de UserQuery."""
        query = UserQuery(query_text="Simple query")
        assert query.analysis_types == []
        assert query.entities == []
        assert query.requires_visualization is False
        assert query.visualization_type is None
        assert query.metadata == {}


class TestAnalysisModels:
    """Testes para modelos de resultado de análise."""

    def test_statistical_summary(self):
        """Testa StatisticalSummary."""
        summary = StatisticalSummary(
            column_name="age",
            data_type="numeric",
            count=100,
            missing_count=5,
            missing_percentage=5.0,
            unique_count=50,
            mean=35.5,
            median=34.0,
            std=12.3,
            min_value=18,
            max_value=65,
            q25=25.0,
            q75=45.0
        )
        assert summary.column_name == "age"
        assert summary.data_type == "numeric"
        assert summary.count == 100
        assert summary.mean == 35.5
        assert summary.median == 34.0
        assert summary.missing_percentage == 5.0

    def test_visualization_result(self):
        """Testa VisualizationResult."""
        viz = VisualizationResult(
            visualization_type=VisualizationType.HISTOGRAM,
            title="Age Distribution",
            description="Distribution of ages in the dataset",
            columns_used=["age"]
        )
        assert viz.visualization_type == VisualizationType.HISTOGRAM
        assert viz.title == "Age Distribution"
        assert "age" in viz.columns_used
        assert viz.metadata == {}

    def test_descriptive_analysis_result(self):
        """Testa DescriptiveAnalysisResult."""
        summary = StatisticalSummary(
            column_name="age",
            data_type="numeric",
            count=100,
            missing_count=0,
            missing_percentage=0.0,
            unique_count=50
        )

        result = DescriptiveAnalysisResult(
            column_summaries=[summary],
            dataset_overview={"rows": 100, "columns": 5}
        )

        assert result.analysis_type == EDAAnalysisType.DESCRIPTIVE
        assert result.status == ProcessingStatus.PENDING
        assert len(result.column_summaries) == 1
        assert result.dataset_overview["rows"] == 100

    def test_pattern_analysis_result(self):
        """Testa PatternAnalysisResult."""
        result = PatternAnalysisResult(
            temporal_patterns={"trend": "increasing"},
            trends=["Upward trend in sales"]
        )
        assert result.analysis_type == EDAAnalysisType.PATTERN
        assert result.temporal_patterns["trend"] == "increasing"
        assert "Upward trend in sales" in result.trends

    def test_anomaly_analysis_result(self):
        """Testa AnomalyAnalysisResult."""
        result = AnomalyAnalysisResult(
            outlier_detection_methods=["IQR", "Z-score"],
            outliers_by_column={"age": [85, 90]},
            recommendations=["Investigate extreme values"]
        )
        assert result.analysis_type == EDAAnalysisType.ANOMALY
        assert "IQR" in result.outlier_detection_methods
        assert result.outliers_by_column["age"] == [85, 90]

    def test_relationship_analysis_result(self):
        """Testa RelationshipAnalysisResult."""
        result = RelationshipAnalysisResult(
            correlation_matrix={"age": {"income": 0.75}},
            strong_correlations=[{"vars": ["age", "income"], "corr": 0.75}]
        )
        assert result.analysis_type == EDAAnalysisType.RELATIONSHIP
        assert result.correlation_matrix["age"]["income"] == 0.75
        assert len(result.strong_correlations) == 1


class TestGraphModels:
    """Testes para modelos de conclusão e resposta final."""

    def test_data_insight(self):
        """Testa DataInsight."""
        insight = DataInsight(
            insight_text="The dataset has a normal distribution of ages",
            confidence=0.85,
            analysis_type=EDAAnalysisType.DESCRIPTIVE,
            importance="high"
        )
        assert insight.confidence == 0.85
        assert insight.analysis_type == EDAAnalysisType.DESCRIPTIVE
        assert insight.importance == "high"

    def test_executive_summary(self):
        """Testa ExecutiveSummary."""
        summary = ExecutiveSummary(
            dataset_name="test_data",
            total_rows=1000,
            total_columns=10,
            key_findings=["Data quality is good", "No major outliers found"]
        )
        assert summary.dataset_name == "test_data"
        assert summary.total_rows == 1000
        assert summary.total_columns == 10
        assert len(summary.key_findings) == 2
        assert isinstance(summary.analysis_timestamp, datetime)

    def test_consolidated_results(self):
        """Testa ConsolidatedResults."""
        insight = DataInsight(
            insight_text="Test insight",
            confidence=0.9,
            analysis_type=EDAAnalysisType.DESCRIPTIVE,
            importance="medium"
        )

        results = ConsolidatedResults(
            query_text="Analyze the data",
            response_text="Analysis completed successfully",
            all_insights=[insight],
            priority_insights=[insight]
        )
        assert results.query_text == "Analyze the data"
        assert len(results.all_insights) == 1
        assert len(results.priority_insights) == 1

    def test_memory_context(self):
        """Testa MemoryContext."""
        insight = DataInsight(
            insight_text="Previous insight",
            confidence=0.8,
            analysis_type=EDAAnalysisType.PATTERN,
            importance="low"
        )

        memory = MemoryContext(
            session_id="test_session_123",
            previous_queries=["What is the mean age?"],
            previous_insights=[insight]
        )
        assert memory.session_id == "test_session_123"
        assert len(memory.previous_queries) == 1
        assert len(memory.previous_insights) == 1
        assert isinstance(memory.created_at, datetime)

    def test_final_response(self):
        """Testa FinalResponse."""
        summary = ExecutiveSummary(
            total_rows=100,
            total_columns=5
        )

        results = ConsolidatedResults(
            query_text="Test query",
            response_text="Test response"
        )

        response = FinalResponse(
            executive_summary=summary,
            consolidated_results=results,
            total_processing_time=5.2
        )
        assert response.total_processing_time == 5.2
        assert response.success is True
        assert len(response.errors) == 0


class TestEDAState:
    """Testes para o estado compartilhado do LangGraph."""

    def test_eda_state_creation(self):
        """Testa criação básica do EDAState."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'name': ['Alice', 'Bob', 'Charlie', 'David']
        })

        state = EDAState(
            csv_data=df,
            user_query="What is the average age?",
            query_classification=["descriptive"]
        )

        assert state.user_query == "What is the average age?"
        assert state.csv_data is not None
        assert len(state.csv_data) == 4
        assert state.query_classification == ["descriptive"]

    def test_eda_state_defaults(self):
        """Testa valores padrão do EDAState."""
        state = EDAState()
        assert state.csv_data is None
        assert state.user_query == ""
        assert state.query_classification == []
        assert state.required_agents == []
        assert state.completed_analyses == set()
        assert state.errors == []

    def test_analysis_completion_methods(self):
        """Testa métodos de controle de análises."""
        state = EDAState()

        # Testa adicionar análise completada
        state.add_completed_analysis(EDAAnalysisType.DESCRIPTIVE)
        assert state.is_analysis_completed(EDAAnalysisType.DESCRIPTIVE)
        assert not state.is_analysis_completed(EDAAnalysisType.PATTERN)

        # Testa múltiplas análises
        state.add_completed_analysis(EDAAnalysisType.PATTERN)
        assert state.is_analysis_completed(EDAAnalysisType.PATTERN)
        assert len(state.completed_analyses) == 2

    def test_error_handling_methods(self):
        """Testa métodos de tratamento de erros."""
        state = EDAState()

        # Estado inicial sem erros
        assert not state.has_errors()
        assert len(state.errors) == 0

        # Adicionar erro
        state.add_error("Test error message")
        assert state.has_errors()
        assert len(state.errors) == 1
        assert "Test error message" in state.errors

        # Adicionar múltiplos erros
        state.add_error("Another error")
        assert len(state.errors) == 2

    def test_analysis_result_methods(self):
        """Testa métodos get/set de resultados de análise."""
        state = EDAState()

        # Inicialmente sem resultados
        assert state.get_analysis_result(EDAAnalysisType.DESCRIPTIVE) is None

        # Definir resultado
        desc_result = DescriptiveAnalysisResult()
        state.set_analysis_result(EDAAnalysisType.DESCRIPTIVE, desc_result)

        # Verificar resultado
        retrieved = state.get_analysis_result(EDAAnalysisType.DESCRIPTIVE)
        assert retrieved is not None
        assert retrieved.analysis_type == EDAAnalysisType.DESCRIPTIVE

        # Testar outros tipos
        pattern_result = PatternAnalysisResult()
        state.set_analysis_result(EDAAnalysisType.PATTERN, pattern_result)
        assert state.get_analysis_result(EDAAnalysisType.PATTERN) is not None


class TestSerialization:
    """Testes de serialização dos modelos."""

    def test_user_query_serialization(self):
        """Testa serialização de UserQuery."""
        query = UserQuery(
            query_text="Test query",
            intent_type=QueryIntentType.DATA_OVERVIEW,
            analysis_types=[EDAAnalysisType.DESCRIPTIVE],
            requires_visualization=True
        )

        # Serializar para dict
        query_dict = query.model_dump()
        assert query_dict["query_text"] == "Test query"
        assert query_dict["intent_type"] == "data_overview"
        assert query_dict["requires_visualization"] is True

        # Deserializar de volta
        query_reconstructed = UserQuery.model_validate(query_dict)
        assert query_reconstructed.query_text == "Test query"
        assert query_reconstructed.intent_type == QueryIntentType.DATA_OVERVIEW

    def test_statistical_summary_serialization(self):
        """Testa serialização de StatisticalSummary."""
        summary = StatisticalSummary(
            column_name="test_column",
            data_type="numeric",
            count=100,
            missing_count=5,
            missing_percentage=5.0,
            unique_count=95
        )

        # Teste de serialização
        summary_dict = summary.model_dump()
        reconstructed = StatisticalSummary.model_validate(summary_dict)
        assert reconstructed.column_name == "test_column"
        assert reconstructed.count == 100

    def test_data_insight_serialization(self):
        """Testa serialização de DataInsight."""
        insight = DataInsight(
            insight_text="Test insight",
            confidence=0.95,
            analysis_type=EDAAnalysisType.DESCRIPTIVE,
            importance="high"
        )

        insight_dict = insight.model_dump()
        reconstructed = DataInsight.model_validate(insight_dict)
        assert reconstructed.confidence == 0.95
        assert reconstructed.analysis_type == EDAAnalysisType.DESCRIPTIVE


class TestModelValidation:
    """Testes de validação dos modelos Pydantic."""

    def test_confidence_score_validation(self):
        """Testa validação de score de confiança."""
        # Score válido
        insight = DataInsight(
            insight_text="Valid insight",
            confidence=0.85,
            analysis_type=EDAAnalysisType.DESCRIPTIVE,
            importance="high"
        )
        assert insight.confidence == 0.85

        # Score inválido deve gerar erro
        with pytest.raises(ValueError):
            DataInsight(
                insight_text="Invalid insight",
                confidence=1.5,  # > 1.0
                analysis_type=EDAAnalysisType.DESCRIPTIVE,
                importance="high"
            )

    def test_required_fields(self):
        """Testa campos obrigatórios."""
        # Deve funcionar com campos obrigatórios
        query = UserQuery(query_text="Required field test")
        assert query.query_text == "Required field test"

        # Deve falhar sem campo obrigatório
        with pytest.raises(ValueError):
            UserQuery()  # query_text é obrigatório