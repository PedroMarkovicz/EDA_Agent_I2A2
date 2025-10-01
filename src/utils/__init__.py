"""
Modulo de utilitarios para o sistema EDA Agent.

Este modulo contem funcoes auxiliares para:
- Manipulacao de arquivos CSV (file_handler)
- Geracao de graficos e visualizacoes (graph_generator)
- Validacao de dados e consultas (validators)
- Formatacao de respostas (formatters)
- Seguranca e validacao de codigo (security)
"""

# File Handler - Manipulacao de arquivos CSV
from .file_handler import (
    CSVFileHandler,
    FileHandlerError,
    get_file_handler,
    read_csv,
    validate_csv_file,
    get_csv_info
)

# Graph Generator - Visualizacoes EDA
from .graph_generator import (
    EDAGraphGenerator,
    GraphGeneratorError,
    get_graph_generator,
    create_histogram,
    create_boxplot,
    create_scatter_plot,
    create_correlation_heatmap
)

# Validators - Validacao de dados e consultas
from .validators import (
    EDADataValidator,
    ValidationError,
    get_data_validator,
    validate_dataframe,
    validate_user_query,
    validate_column_for_analysis,
    is_dataframe_valid,
    is_query_safe
)

# Formatters - Formatacao de respostas
from .formatters import (
    EDAResponseFormatter,
    FormatterError,
    get_response_formatter,
    format_descriptive_analysis,
    format_data_summary,
    format_statistics_table
)

# Security - Validacao de codigo e seguranca
from .security import (
    CodeSecurityValidator,
    SecurityError,
    get_security_validator,
    validate_python_code,
    execute_code_safely,
    sanitize_user_input,
    is_code_safe,
    is_file_path_safe
)

__all__ = [
    # File Handler
    "CSVFileHandler",
    "FileHandlerError",
    "get_file_handler",
    "read_csv",
    "validate_csv_file",
    "get_csv_info",

    # Graph Generator
    "EDAGraphGenerator",
    "GraphGeneratorError",
    "get_graph_generator",
    "create_histogram",
    "create_boxplot",
    "create_scatter_plot",
    "create_correlation_heatmap",

    # Validators
    "EDADataValidator",
    "ValidationError",
    "get_data_validator",
    "validate_dataframe",
    "validate_user_query",
    "validate_column_for_analysis",
    "is_dataframe_valid",
    "is_query_safe",

    # Formatters
    "EDAResponseFormatter",
    "FormatterError",
    "get_response_formatter",
    "format_descriptive_analysis",
    "format_data_summary",
    "format_statistics_table",

    # Security
    "CodeSecurityValidator",
    "SecurityError",
    "get_security_validator",
    "validate_python_code",
    "execute_code_safely",
    "sanitize_user_input",
    "is_code_safe",
    "is_file_path_safe"
]

# Versao do modulo utils
__version__ = "1.0.0"