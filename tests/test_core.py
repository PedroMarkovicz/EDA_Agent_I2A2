"""
Testes de integracao para componentes core do sistema EDA.
Testa Config, LLMManager, CSVProcessor, QueryInterpreter,
CodeExecutor, MemoryManager e Logger.
"""

import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.core.config import Config, get_config, reset_config
from src.core.llm_manager import LLMManager, get_llm_manager
from src.core.csv_processor import CSVProcessor, get_csv_processor
from src.core.query_interpreter import QueryInterpreter, get_query_interpreter
from src.core.code_executor import CodeExecutor, get_code_executor
from src.core.memory_manager import MemoryManager, get_memory_manager
from src.core.logger import get_logger
from src.models.enums import QueryIntentType, EDAAnalysisType


class TestConfig:
    """Testes para Config."""

    def test_initialization(self, mock_env_vars):
        """Testa inicializacao do Config."""
        reset_config()
        config = Config()
        assert config is not None
        assert config.openai_api_key == "test-api-key"

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_attributes(self, mock_env_vars):
        """Testa atributos de configuracao."""
        reset_config()
        config = Config()
        assert hasattr(config, 'llm_model')
        assert hasattr(config, 'openai_api_key')
        assert hasattr(config, 'log_level')

    def test_reset_config(self, mock_env_vars):
        """Testa reset do singleton."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2


class TestLLMManager:
    """Testes para LLMManager."""

    def test_initialization(self):
        """Testa inicializacao do LLMManager."""
        with patch('openai.api_key'):
            manager = LLMManager()
            assert manager is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        manager1 = get_llm_manager()
        manager2 = get_llm_manager()
        assert manager1 is manager2

    @patch('openai.chat.completions.create')
    def test_chat_completion_success(self, mock_create):
        """Testa chat completion com sucesso."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        mock_create.return_value = mock_response

        manager = get_llm_manager()
        result = manager.chat_completion([{"role": "user", "content": "Test"}])

        assert result["content"] == "Test response"
        assert result["usage"]["total_tokens"] == 30

    @patch('openai.chat.completions.create')
    def test_chat_completion_with_retry(self, mock_create):
        """Testa retry em caso de falha."""
        mock_create.side_effect = [Exception("API Error"), Exception("API Error")]

        manager = get_llm_manager()

        with pytest.raises(Exception):
            manager.chat_completion([{"role": "user", "content": "Test"}])

        assert mock_create.call_count == 3

    def test_usage_stats(self):
        """Testa estatisticas de uso."""
        manager = get_llm_manager()
        stats = manager.get_usage_stats()

        assert "total_requests" in stats
        assert "total_tokens" in stats
        assert "total_cost" in stats

    def test_rate_limiting(self):
        """Testa rate limiting."""
        manager = LLMManager()
        manager.rate_limit = 2
        manager._check_rate_limit()
        assert len(manager.request_times) <= manager.rate_limit


class TestCSVProcessor:
    """Testes para CSVProcessor."""

    def test_initialization(self):
        """Testa inicializacao do CSVProcessor."""
        processor = CSVProcessor()
        assert processor is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        processor1 = get_csv_processor()
        processor2 = get_csv_processor()
        assert processor1 is processor2

    def test_validate_csv_valid(self, simple_df):
        """Testa validacao de CSV valido."""
        processor = CSVProcessor()
        is_valid, errors = processor.validate_csv(simple_df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_csv_empty(self):
        """Testa validacao de CSV vazio."""
        processor = CSVProcessor()
        empty_df = pd.DataFrame()
        is_valid, errors = processor.validate_csv(empty_df)

        assert is_valid is False
        assert len(errors) > 0

    def test_process_csv_from_string(self, csv_string_simple):
        """Testa processamento de CSV a partir de string."""
        processor = CSVProcessor()
        df, metadata = processor.process_csv_string(csv_string_simple)

        assert df is not None
        assert len(df) == 3
        assert len(df.columns) == 3
        assert metadata is not None

    def test_generate_metadata(self, simple_df):
        """Testa geracao de metadata."""
        processor = CSVProcessor()
        metadata = processor.generate_metadata(simple_df)

        assert "shape" in metadata
        assert "columns" in metadata
        assert "dtypes" in metadata
        assert metadata["shape"] == (3, 3)

    def test_detect_column_types(self, simple_df):
        """Testa deteccao de tipos de coluna."""
        processor = CSVProcessor()
        types = processor.detect_column_types(simple_df)

        assert "nome" in types
        assert "idade" in types
        assert "salario" in types


class TestQueryInterpreter:
    """Testes para QueryInterpreter."""

    def test_initialization(self):
        """Testa inicializacao do QueryInterpreter."""
        interpreter = QueryInterpreter()
        assert interpreter is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        interpreter1 = get_query_interpreter()
        interpreter2 = get_query_interpreter()
        assert interpreter1 is interpreter2

    def test_classify_query_descriptive(self, csv_metadata_simple):
        """Testa classificacao de query descritiva."""
        interpreter = QueryInterpreter()

        with patch.object(interpreter.llm_manager, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "content": '{"intent_type": "statistical_summary", "analysis_types": ["descriptive"]}',
                "usage": {"total_tokens": 100}
            }

            result = interpreter.classify_query(
                "Qual e a media de idade?",
                csv_metadata_simple
            )

            assert result is not None
            assert "intent_type" in result

    def test_extract_entities(self):
        """Testa extracao de entidades."""
        interpreter = QueryInterpreter()
        entities = interpreter._extract_entities("Qual e o valor maximo da coluna idade?")

        assert isinstance(entities, list)

    def test_classify_intent(self):
        """Testa classificacao de intencao."""
        interpreter = QueryInterpreter()
        intent = interpreter._classify_intent("Mostre a distribuicao de idades")

        assert intent in QueryIntentType or intent is None

    def test_determine_analysis_types(self):
        """Testa determinacao de tipos de analise."""
        interpreter = QueryInterpreter()
        types = interpreter._determine_analysis_types(
            "Analise descritiva dos dados",
            QueryIntentType.STATISTICAL_SUMMARY
        )

        assert isinstance(types, list)
        assert EDAAnalysisType.DESCRIPTIVE in types

    def test_detect_visualization_needs(self):
        """Testa deteccao de necessidade de visualizacao."""
        interpreter = QueryInterpreter()
        needs_viz, viz_type = interpreter._detect_visualization_needs(
            "Mostre um histograma de idades"
        )

        assert isinstance(needs_viz, bool)


class TestCodeExecutor:
    """Testes para CodeExecutor."""

    def test_initialization(self):
        """Testa inicializacao do CodeExecutor."""
        executor = CodeExecutor()
        assert executor is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        executor1 = get_code_executor()
        executor2 = get_code_executor()
        assert executor1 is executor2

    def test_execute_safe_code(self):
        """Testa execucao de codigo seguro."""
        executor = CodeExecutor()
        code = "result = 2 + 2"
        result, error = executor.execute_code(code, {"result": None})

        assert error is None
        assert result["result"] == 4

    def test_execute_with_dataframe(self, simple_df):
        """Testa execucao com dataframe."""
        executor = CodeExecutor()
        code = "mean_age = df['idade'].mean()"
        context = {"df": simple_df}
        result, error = executor.execute_code(code, context)

        assert error is None
        assert "mean_age" in result

    def test_block_unsafe_code(self):
        """Testa bloqueio de codigo inseguro."""
        executor = CodeExecutor()
        unsafe_code = "import os; os.system('ls')"

        is_safe = executor._is_safe_code(unsafe_code)
        assert is_safe is False

    def test_validate_imports(self):
        """Testa validacao de imports."""
        executor = CodeExecutor()

        safe_imports = "import pandas as pd\nimport numpy as np"
        assert executor._is_safe_code(safe_imports) is True

        unsafe_imports = "import subprocess"
        assert executor._is_safe_code(unsafe_imports) is False


class TestMemoryManager:
    """Testes para MemoryManager."""

    def test_initialization(self):
        """Testa inicializacao do MemoryManager."""
        manager = MemoryManager()
        assert manager is not None

    def test_singleton_pattern(self):
        """Testa padrao singleton."""
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()
        assert manager1 is manager2

    def test_save_and_retrieve_insight(self):
        """Testa salvamento e recuperacao de insight."""
        manager = MemoryManager()
        session_id = "test_session"
        insight = {"text": "Test insight", "confidence": 0.9}

        manager.save_insight(session_id, insight)
        insights = manager.get_session_insights(session_id)

        assert len(insights) > 0
        assert insights[0]["text"] == "Test insight"

    def test_save_and_retrieve_query(self):
        """Testa salvamento e recuperacao de query."""
        manager = MemoryManager()
        session_id = "test_session"
        query = "Test query"

        manager.save_query(session_id, query)
        queries = manager.get_session_queries(session_id)

        assert len(queries) > 0
        assert query in queries

    def test_clear_session(self):
        """Testa limpeza de sessao."""
        manager = MemoryManager()
        session_id = "test_session"

        manager.save_insight(session_id, {"text": "Test"})
        manager.clear_session(session_id)
        insights = manager.get_session_insights(session_id)

        assert len(insights) == 0

    def test_get_session_context(self):
        """Testa recuperacao de contexto de sessao."""
        manager = MemoryManager()
        session_id = "test_session"

        manager.save_query(session_id, "Query 1")
        manager.save_insight(session_id, {"text": "Insight 1"})

        context = manager.get_session_context(session_id)

        assert "queries" in context
        assert "insights" in context
        assert len(context["queries"]) > 0


class TestLogger:
    """Testes para Logger."""

    def test_get_logger(self):
        """Testa obtencao de logger."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_logger_info(self, caplog):
        """Testa logging de info."""
        logger = get_logger("test_logger")
        logger.info("Test info message")

        assert "Test info message" in caplog.text

    def test_logger_error(self, caplog):
        """Testa logging de erro."""
        logger = get_logger("test_logger")
        logger.error("Test error message")

        assert "Test error message" in caplog.text

    def test_logger_warning(self, caplog):
        """Testa logging de warning."""
        logger = get_logger("test_logger")
        logger.warning("Test warning message")

        assert "Test warning message" in caplog.text