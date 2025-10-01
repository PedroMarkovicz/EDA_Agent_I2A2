"""
Fixtures compartilhados para testes do sistema EDA.
Fornece mocks, dataframes de teste e configuracoes comuns.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from src.models.enums import EDAAnalysisType, QueryIntentType, ProcessingStatus
from src.models.query_schema import UserQuery, QueryClassification
from src.models.analysis_result import (
    DescriptiveAnalysisResult,
    PatternAnalysisResult,
    AnomalyAnalysisResult,
    RelationshipAnalysisResult,
    StatisticalSummary
)


# ============================================================================
# FIXTURES DE DATAFRAMES
# ============================================================================

@pytest.fixture
def simple_df():
    """DataFrame simples para testes basicos."""
    return pd.DataFrame({
        'nome': ['Ana', 'Bruno', 'Carlos'],
        'idade': [25, 30, 28],
        'salario': [3000.0, 4500.0, 3800.0]
    })


@pytest.fixture
def numeric_df():
    """DataFrame totalmente numerico."""
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10, 20, 30, 40, 50],
        'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
    })


@pytest.fixture
def temporal_df():
    """DataFrame com dados temporais."""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'data': dates,
        'vendas': np.random.randint(100, 500, 30),
        'temperatura': np.random.uniform(15, 35, 30)
    })


@pytest.fixture
def outliers_df():
    """DataFrame com outliers claros."""
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 95)
    outliers = [200, 250, 10, 5, 300]
    return pd.DataFrame({
        'valores': np.concatenate([normal_data, outliers]),
        'categoria': ['A'] * 50 + ['B'] * 50
    })


@pytest.fixture
def correlation_df():
    """DataFrame com correlacoes fortes."""
    np.random.seed(42)
    x = np.random.uniform(0, 100, 50)
    return pd.DataFrame({
        'x': x,
        'y_high_corr': x * 2 + np.random.normal(0, 5, 50),
        'y_low_corr': np.random.uniform(0, 100, 50),
        'z': x * -1.5 + np.random.normal(0, 10, 50)
    })


@pytest.fixture
def categorical_df():
    """DataFrame com dados categoricos."""
    return pd.DataFrame({
        'departamento': ['TI', 'Vendas', 'TI', 'Marketing', 'Vendas'] * 10,
        'nivel': ['Junior', 'Pleno', 'Senior', 'Pleno', 'Junior'] * 10,
        'salario': np.random.randint(3000, 10000, 50)
    })


@pytest.fixture
def missing_data_df():
    """DataFrame com dados ausentes."""
    df = pd.DataFrame({
        'col1': [1, 2, None, 4, 5, None, 7, 8, 9, 10],
        'col2': [10, None, 30, None, 50, 60, None, 80, 90, 100],
        'col3': ['A', 'B', 'C', None, 'E', 'F', 'G', None, 'I', 'J']
    })
    return df


# ============================================================================
# FIXTURES DE MOCKS - LLM
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock de resposta do LLM."""
    return {
        "content": "Resposta simulada do LLM para teste",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        },
        "model": "gpt-5-nano-2025-08-07",
        "duration": 1.5,
        "estimated_cost": 0.0001
    }


@pytest.fixture
def mock_llm_manager(mock_llm_response):
    """Mock do LLMManager."""
    mock = Mock()
    mock.chat_completion.return_value = mock_llm_response
    mock.generate_text.return_value = mock_llm_response["content"]
    mock.get_usage_stats.return_value = {
        "total_requests": 1,
        "total_tokens": 150,
        "total_cost": 0.0001
    }
    return mock


# ============================================================================
# FIXTURES DE QUERIES
# ============================================================================

@pytest.fixture
def sample_user_query():
    """Query de usuario simples."""
    return UserQuery(
        query_text="Qual e a media de idade?",
        intent_type=QueryIntentType.STATISTICAL_SUMMARY,
        analysis_types=[EDAAnalysisType.DESCRIPTIVE],
        requires_visualization=False
    )


@pytest.fixture
def sample_query_classification(sample_user_query):
    """Classificacao de query de exemplo."""
    return QueryClassification(
        query=sample_user_query,
        confidence_score=0.85,
        required_agents=[EDAAnalysisType.DESCRIPTIVE],
        processing_order=[EDAAnalysisType.DESCRIPTIVE],
        estimated_complexity="low"
    )


# ============================================================================
# FIXTURES DE RESULTADOS DE ANALISE
# ============================================================================

@pytest.fixture
def sample_statistical_summary():
    """Resumo estatistico de exemplo."""
    return StatisticalSummary(
        column_name="idade",
        data_type="integer",
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


@pytest.fixture
def sample_descriptive_result(sample_statistical_summary):
    """Resultado de analise descritiva de exemplo."""
    return DescriptiveAnalysisResult(
        analysis_type=EDAAnalysisType.DESCRIPTIVE,
        status=ProcessingStatus.COMPLETED,
        column_summaries=[sample_statistical_summary],
        dataset_overview={
            "total_rows": 100,
            "total_columns": 5,
            "completeness_percentage": 95.0
        },
        insights=["Dataset possui boa qualidade de dados"]
    )


# ============================================================================
# FIXTURES DE CONFIGURACAO
# ============================================================================

@pytest.fixture
def test_config():
    """Configuracao de teste."""
    return {
        "llm_model": "gpt-5-nano-2025-08-07",
        "max_tokens": 1000,
        "temperature": 0.7,
        "log_level": "INFO"
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock de variaveis de ambiente."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-langchain-key")
    monkeypatch.setenv("LOG_LEVEL", "INFO")


# ============================================================================
# FIXTURES DE CSV
# ============================================================================

@pytest.fixture
def csv_string_simple():
    """String CSV simples."""
    return "nome,idade,salario\nAna,25,3000\nBruno,30,4500\nCarlos,28,3800"


@pytest.fixture
def csv_metadata_simple():
    """Metadata de CSV simples."""
    return {
        "shape": (3, 3),
        "columns": ["nome", "idade", "salario"],
        "dtypes": {"nome": "object", "idade": "int64", "salario": "float64"},
        "memory_usage_mb": 0.001
    }


# ============================================================================
# FIXTURES DE ESTADO
# ============================================================================

@pytest.fixture
def empty_state():
    """Estado vazio do LangGraph."""
    from src.graph.state import EDAState
    return EDAState()


@pytest.fixture
def populated_state(simple_df, sample_query_classification):
    """Estado populado com dados de teste."""
    from src.graph.state import EDAState
    state = EDAState(
        csv_data=simple_df,
        user_query="Teste query",
        query_classification_result=sample_query_classification
    )
    state.required_agents = ["data_analyzer"]
    return state