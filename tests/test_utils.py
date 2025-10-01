"""
Testes para funcoes auxiliares do sistema EDA.
Testa FileHandler, GraphGenerator, Validators, Formatters e Security.
"""

import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch
from src.utils.file_handler import FileHandler, get_file_handler
from src.utils.graph_generator import GraphGenerator, get_graph_generator
from src.utils.validators import Validators, get_validators
from src.utils.formatters import Formatters, get_formatters
from src.utils.security import Security, get_security
from src.models.enums import VisualizationType


class TestFileHandler:
    """Testes para FileHandler."""

    def test_initialization(self):
        """Testa inicializacao."""
        handler = FileHandler()
        assert handler is not None

    def test_singleton_pattern(self):
        """Testa singleton."""
        h1 = get_file_handler()
        h2 = get_file_handler()
        assert h1 is h2

    def test_read_csv_from_path(self, tmp_path):
        """Testa leitura de CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

        handler = FileHandler()
        df = handler.read_csv(str(csv_file))

        assert df is not None
        assert len(df) == 2
        assert len(df.columns) == 3

    def test_read_nonexistent_file(self):
        """Testa leitura de arquivo inexistente."""
        handler = FileHandler()

        with pytest.raises(FileNotFoundError):
            handler.read_csv("nonexistent.csv")


class TestGraphGenerator:
    """Testes para GraphGenerator."""

    def test_initialization(self):
        """Testa inicializacao."""
        generator = GraphGenerator()
        assert generator is not None

    def test_singleton_pattern(self):
        """Testa singleton."""
        g1 = get_graph_generator()
        g2 = get_graph_generator()
        assert g1 is g2

    def test_generate_histogram(self, numeric_df):
        """Testa geracao de histograma."""
        generator = GraphGenerator()
        result = generator.generate_histogram(numeric_df, 'col1', title="Test Histogram")

        assert result is not None
        assert result.visualization_type == VisualizationType.HISTOGRAM
        assert result.title == "Test Histogram"

    def test_generate_boxplot(self, numeric_df):
        """Testa geracao de boxplot."""
        generator = GraphGenerator()
        result = generator.generate_boxplot(numeric_df, ['col1', 'col2'], title="Test Boxplot")

        assert result is not None
        assert result.visualization_type == VisualizationType.BOXPLOT

    def test_generate_scatter_plot(self, correlation_df):
        """Testa geracao de scatter plot."""
        generator = GraphGenerator()
        result = generator.generate_scatter_plot(
            correlation_df, 'idade', 'salario', title="Test Scatter"
        )

        assert result is not None
        assert result.visualization_type == VisualizationType.SCATTER_PLOT

    def test_generate_correlation_heatmap(self, correlation_df):
        """Testa geracao de heatmap."""
        generator = GraphGenerator()
        result = generator.generate_correlation_heatmap(
            correlation_df, title="Test Heatmap"
        )

        assert result is not None
        assert result.visualization_type == VisualizationType.CORRELATION_HEATMAP


class TestValidators:
    """Testes para Validators."""

    def test_initialization(self):
        """Testa inicializacao."""
        validators = Validators()
        assert validators is not None

    def test_singleton_pattern(self):
        """Testa singleton."""
        v1 = get_validators()
        v2 = get_validators()
        assert v1 is v2

    def test_validate_dataframe_valid(self, simple_df):
        """Testa validacao de dataframe valido."""
        validators = Validators()
        is_valid, errors = validators.validate_dataframe(simple_df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_dataframe_empty(self):
        """Testa validacao de dataframe vazio."""
        validators = Validators()
        empty_df = pd.DataFrame()
        is_valid, errors = validators.validate_dataframe(empty_df)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_query_valid(self):
        """Testa validacao de query valida."""
        validators = Validators()
        is_valid, error = validators.validate_query("Qual e a media de idade?")

        assert is_valid is True
        assert error is None

    def test_validate_query_empty(self):
        """Testa validacao de query vazia."""
        validators = Validators()
        is_valid, error = validators.validate_query("")

        assert is_valid is False
        assert error is not None

    def test_validate_column_exists(self, simple_df):
        """Testa validacao de existencia de coluna."""
        validators = Validators()

        assert validators.validate_column_exists(simple_df, 'idade') is True
        assert validators.validate_column_exists(simple_df, 'inexistente') is False


class TestFormatters:
    """Testes para Formatters."""

    def test_initialization(self):
        """Testa inicializacao."""
        formatters = Formatters()
        assert formatters is not None

    def test_singleton_pattern(self):
        """Testa singleton."""
        f1 = get_formatters()
        f2 = get_formatters()
        assert f1 is f2

    def test_format_number(self):
        """Testa formatacao de numeros."""
        formatters = Formatters()

        assert formatters.format_number(1234.567) == "1,234.57"
        assert formatters.format_number(1000) == "1,000"

    def test_format_percentage(self):
        """Testa formatacao de percentuais."""
        formatters = Formatters()

        assert formatters.format_percentage(0.8532) == "85.32%"
        assert formatters.format_percentage(1.0) == "100.00%"

    def test_format_response_text(self):
        """Testa formatacao de texto de resposta."""
        formatters = Formatters()
        insights = ["Insight 1", "Insight 2", "Insight 3"]

        formatted = formatters.format_response_text(
            query="Test query",
            insights=insights
        )

        assert formatted is not None
        assert isinstance(formatted, str)
        assert "Insight 1" in formatted


class TestSecurity:
    """Testes para Security."""

    def test_initialization(self):
        """Testa inicializacao."""
        security = Security()
        assert security is not None

    def test_singleton_pattern(self):
        """Testa singleton."""
        s1 = get_security()
        s2 = get_security()
        assert s1 is s2

    def test_sanitize_input_safe(self):
        """Testa sanitizacao de entrada segura."""
        security = Security()
        safe_input = "Qual e a media de idade?"

        sanitized = security.sanitize_input(safe_input)
        assert sanitized == safe_input

    def test_sanitize_input_dangerous(self):
        """Testa sanitizacao de entrada perigosa."""
        security = Security()
        dangerous_input = "<script>alert('xss')</script>"

        sanitized = security.sanitize_input(dangerous_input)
        assert "<script>" not in sanitized

    def test_validate_code_safety(self):
        """Testa validacao de seguranca de codigo."""
        security = Security()

        safe_code = "import pandas as pd\ndf.head()"
        assert security.is_safe_code(safe_code) is True

        unsafe_code = "import os\nos.system('rm -rf /')"
        assert security.is_safe_code(unsafe_code) is False

    def test_block_dangerous_imports(self):
        """Testa bloqueio de imports perigosos."""
        security = Security()

        assert security.is_safe_code("import subprocess") is False
        assert security.is_safe_code("import sys") is False
        assert security.is_safe_code("import pandas") is True