"""
Processador centralizado para arquivos CSV.
Realiza validação, carregamento e pré-processamento de dados CSV,
respeitando limites de tamanho e extensões configurados no .env,
com detecção automática de encoding e estrutura.
"""

import pandas as pd
import io
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mimetypes

from .config import get_config
from .logger import get_logger, log_data_operation, log_error_with_context, log_security_event


class CSVValidationError(Exception):
    """Erro de validação específico para arquivos CSV."""
    pass


class CSVProcessor:
    """Processador centralizado para arquivos CSV."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("csv_processor")
        self.allowed_extensions = self.config.get_allowed_extensions()
        self.max_size_bytes = self.config.max_upload_size_mb * 1024 * 1024

    def validate_file(self, file_path: str, file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Valida arquivo CSV antes do processamento.

        Args:
            file_path: Caminho do arquivo
            file_content: Conteúdo do arquivo em bytes (opcional)

        Returns:
            Dict com informações de validação

        Raises:
            CSVValidationError: Se arquivo não atende critérios
        """
        path = Path(file_path)
        validation_result = {
            "valid": False,
            "file_size": 0,
            "extension": path.suffix.lower(),
            "mime_type": None,
            "encoding": None,
            "errors": []
        }

        try:
            # Validar extensão
            if validation_result["extension"][1:] not in self.allowed_extensions:
                validation_result["errors"].append(
                    f"Extensão '{validation_result['extension']}' não permitida. "
                    f"Permitidas: {', '.join(self.allowed_extensions)}"
                )

            # Validar tamanho
            if file_content:
                validation_result["file_size"] = len(file_content)
            elif path.exists():
                validation_result["file_size"] = path.stat().st_size

            if validation_result["file_size"] > self.max_size_bytes:
                size_mb = validation_result["file_size"] / (1024 * 1024)
                validation_result["errors"].append(
                    f"Arquivo muito grande ({size_mb:.1f}MB). "
                    f"Máximo permitido: {self.config.max_upload_size_mb}MB"
                )

            # Validar tipo MIME
            mime_type, _ = mimetypes.guess_type(file_path)
            validation_result["mime_type"] = mime_type

            if mime_type not in ["text/csv", "text/plain", None]:
                log_security_event(
                    "suspicious_file_type",
                    {"file_path": file_path, "mime_type": mime_type}
                )

            # Detectar encoding se conteúdo disponível
            if file_content:
                if HAS_CHARDET:
                    detected = chardet.detect(file_content)
                    validation_result["encoding"] = detected.get("encoding", "utf-8")
                else:
                    validation_result["encoding"] = "utf-8"

            # Validação bem-sucedida se não há erros
            validation_result["valid"] = len(validation_result["errors"]) == 0

            log_data_operation(
                "file_validation",
                {
                    "file_path": file_path,
                    "valid": validation_result["valid"],
                    "size_mb": validation_result["file_size"] / (1024 * 1024),
                    "errors_count": len(validation_result["errors"])
                }
            )

            return validation_result

        except Exception as e:
            log_error_with_context(e, {"file_path": file_path, "operation": "validation"})
            validation_result["errors"].append(f"Erro na validação: {e}")
            return validation_result

    def detect_encoding(self, file_content: bytes) -> str:
        """
        Detecta encoding do arquivo.

        Args:
            file_content: Conteúdo do arquivo em bytes

        Returns:
            Encoding detectado
        """
        try:
            if HAS_CHARDET:
                result = chardet.detect(file_content)
                encoding = result.get("encoding", "utf-8")
                confidence = result.get("confidence", 0)
            else:
                encoding = "utf-8"
                confidence = 1.0

            self.logger.info(f"Encoding detectado: {encoding} (confiança: {confidence:.2f})")

            # Fallback para encodings comuns se confiança baixa
            if confidence < 0.7:
                common_encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
                for enc in common_encodings:
                    try:
                        file_content.decode(enc)
                        self.logger.info(f"Usando encoding fallback: {enc}")
                        return enc
                    except UnicodeDecodeError:
                        continue

            return encoding

        except Exception as e:
            log_error_with_context(e, {"operation": "encoding_detection"})
            return "utf-8"

    def load_csv(
        self,
        file_path: str,
        file_content: Optional[bytes] = None,
        encoding: Optional[str] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Carrega arquivo CSV como DataFrame.

        Args:
            file_path: Caminho do arquivo
            file_content: Conteúdo em bytes (opcional)
            encoding: Encoding específico (opcional)
            **kwargs: Parâmetros adicionais para pd.read_csv

        Returns:
            Tuple com DataFrame e metadados

        Raises:
            CSVValidationError: Se arquivo inválido
        """
        # Validar arquivo
        validation = self.validate_file(file_path, file_content)
        if not validation["valid"]:
            raise CSVValidationError(f"Arquivo inválido: {'; '.join(validation['errors'])}")

        try:
            # Determinar encoding
            if file_content and not encoding:
                encoding = self.detect_encoding(file_content)
            elif not encoding:
                encoding = "utf-8"

            # Configurações padrão para leitura
            read_kwargs = {
                "encoding": encoding,
                "low_memory": False,
                "skipinitialspace": True,
                **kwargs
            }

            # Carregar DataFrame
            if file_content:
                # Ler de bytes
                csv_string = file_content.decode(encoding)
                df = pd.read_csv(io.StringIO(csv_string), **read_kwargs)
            else:
                # Ler de arquivo
                df = pd.read_csv(file_path, **read_kwargs)

            # Gerar metadados
            metadata = self.generate_metadata(df, file_path, validation)

            log_data_operation(
                "csv_loaded",
                {
                    "file_path": file_path,
                    "shape": df.shape,
                    "encoding": encoding,
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
                }
            )

            return df, metadata

        except UnicodeDecodeError as e:
            error_msg = f"Erro de encoding: {e}. Tente especificar encoding diferente."
            log_error_with_context(e, {"file_path": file_path, "encoding": encoding})
            raise CSVValidationError(error_msg)

        except pd.errors.EmptyDataError:
            error_msg = "Arquivo CSV está vazio ou não contém dados válidos."
            raise CSVValidationError(error_msg)

        except pd.errors.ParserError as e:
            error_msg = f"Erro na estrutura do CSV: {e}"
            log_error_with_context(e, {"file_path": file_path})
            raise CSVValidationError(error_msg)

        except Exception as e:
            log_error_with_context(e, {"file_path": file_path, "operation": "csv_load"})
            raise CSVValidationError(f"Erro inesperado ao carregar CSV: {e}")

    def generate_metadata(
        self,
        df: pd.DataFrame,
        file_path: str,
        validation_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera metadados completos do DataFrame.

        Args:
            df: DataFrame carregado
            file_path: Caminho original do arquivo
            validation_info: Informações de validação

        Returns:
            Dict com metadados detalhados
        """
        try:
            # Informações básicas
            basic_info = {
                "file_path": file_path,
                "shape": df.shape,
                "columns": list(df.columns),
                "size_mb": validation_info.get("file_size", 0) / (1024 * 1024),
                "encoding": validation_info.get("encoding", "unknown")
            }

            # Análise de tipos de dados
            dtypes_info = {
                "data_types": df.dtypes.astype(str).to_dict(),
                "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
                "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
                "datetime_columns": list(df.select_dtypes(include=["datetime"]).columns)
            }

            # Análise de qualidade dos dados
            quality_info = {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "unique_values_per_column": {col: df[col].nunique() for col in df.columns}
            }

            # Estatísticas rápidas para colunas numéricas
            numeric_stats = {}
            for col in dtypes_info["numeric_columns"]:
                try:
                    numeric_stats[col] = {
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max())
                    }
                except Exception:
                    numeric_stats[col] = {"error": "Não foi possível calcular estatísticas"}

            # Amostra dos dados (primeiras linhas)
            sample_data = df.head(5).to_dict('records')

            metadata = {
                **basic_info,
                **dtypes_info,
                "quality": quality_info,
                "numeric_stats": numeric_stats,
                "sample_data": sample_data,
                "memory_usage": {
                    "total_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                    "per_column": (df.memory_usage(deep=True) / (1024 * 1024)).to_dict()
                }
            }

            return metadata

        except Exception as e:
            log_error_with_context(e, {"operation": "metadata_generation"})
            return {"error": f"Erro ao gerar metadados: {e}"}

    def get_column_info(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Retorna informações detalhadas sobre uma coluna específica.

        Args:
            df: DataFrame
            column: Nome da coluna

        Returns:
            Dict com informações da coluna
        """
        if column not in df.columns:
            return {"error": f"Coluna '{column}' não encontrada"}

        try:
            col_data = df[column]
            info = {
                "name": column,
                "dtype": str(col_data.dtype),
                "count": int(col_data.count()),
                "missing": int(col_data.isnull().sum()),
                "unique": int(col_data.nunique()),
                "most_frequent": None,
                "frequency": None
            }

            # Valor mais frequente
            if not col_data.empty:
                value_counts = col_data.value_counts()
                if not value_counts.empty:
                    info["most_frequent"] = value_counts.index[0]
                    info["frequency"] = int(value_counts.iloc[0])

            # Estatísticas para colunas numéricas
            if pd.api.types.is_numeric_dtype(col_data):
                info.update({
                    "mean": float(col_data.mean()) if not col_data.empty else None,
                    "median": float(col_data.median()) if not col_data.empty else None,
                    "std": float(col_data.std()) if not col_data.empty else None,
                    "min": float(col_data.min()) if not col_data.empty else None,
                    "max": float(col_data.max()) if not col_data.empty else None,
                    "quantiles": {
                        "25%": float(col_data.quantile(0.25)) if not col_data.empty else None,
                        "75%": float(col_data.quantile(0.75)) if not col_data.empty else None
                    }
                })

            return info

        except Exception as e:
            log_error_with_context(e, {"column": column, "operation": "column_info"})
            return {"error": f"Erro ao analisar coluna: {e}"}

    def validate_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida a estrutura do DataFrame carregado.

        Args:
            df: DataFrame para validar

        Returns:
            Dicionário com informações de validação
        """
        try:
            validation_info = {
                "is_valid": True,
                "shape": df.shape,
                "columns_count": len(df.columns),
                "rows_count": len(df),
                "duplicate_columns": [],
                "empty_columns": [],
                "data_types": {},
                "memory_usage_mb": 0.0,
                "warnings": []
            }

            # Verificar colunas duplicadas
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                validation_info["duplicate_columns"] = duplicate_cols
                validation_info["warnings"].append(f"Colunas duplicadas encontradas: {duplicate_cols}")

            # Verificar colunas vazias
            empty_cols = [col for col in df.columns if df[col].isna().all()]
            if empty_cols:
                validation_info["empty_columns"] = empty_cols
                validation_info["warnings"].append(f"Colunas completamente vazias: {empty_cols}")

            # Analisar tipos de dados
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                validation_info["data_types"][col] = dtype_str

            # Calcular uso de memória
            memory_usage = df.memory_usage(deep=True).sum()
            validation_info["memory_usage_mb"] = memory_usage / (1024 * 1024)

            # Verificar se o DataFrame é muito grande
            if validation_info["memory_usage_mb"] > 500:  # 500MB
                validation_info["warnings"].append(
                    f"Dataset grande ({validation_info['memory_usage_mb']:.1f}MB), "
                    "operações podem ser lentas"
                )

            log_data_operation(
                "csv_structure_validation",
                validation_info,
                "csv_processor"
            )

            return validation_info

        except Exception as e:
            log_error_with_context(
                e,
                {"dataframe_shape": df.shape if df is not None else "unknown"},
                "csv_structure_validation_error"
            )
            return {
                "is_valid": False,
                "error": str(e),
                "warnings": [f"Erro na validação da estrutura: {str(e)}"]
            }

    def suggest_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Sugere otimizações de tipos de dados.

        Args:
            df: DataFrame a ser analisado

        Returns:
            Dict com sugestões de tipos
        """
        suggestions = {}

        for column in df.columns:
            current_dtype = str(df[column].dtype)
            col_data = df[column].dropna()

            if col_data.empty:
                continue

            try:
                # Sugestões para colunas object
                if current_dtype == "object":
                    # Verificar se pode ser categoria
                    unique_ratio = col_data.nunique() / len(col_data)
                    if unique_ratio < 0.5:
                        suggestions[column] = "category"

                    # Verificar se pode ser numérico
                    elif col_data.dtype == "object":
                        try:
                            pd.to_numeric(col_data)
                            suggestions[column] = "numeric"
                        except (ValueError, TypeError):
                            pass

                    # Verificar se pode ser datetime
                    try:
                        pd.to_datetime(col_data.head(100))
                        suggestions[column] = "datetime"
                    except (ValueError, TypeError):
                        pass

                # Sugestões para colunas numéricas
                elif pd.api.types.is_numeric_dtype(col_data):
                    if col_data.min() >= 0 and col_data.max() <= 255 and col_data.dtype != "uint8":
                        suggestions[column] = "uint8"
                    elif col_data.dtype == "float64" and col_data.astype(int).equals(col_data):
                        suggestions[column] = "int64"

            except Exception as e:
                self.logger.warning(f"Erro ao analisar tipo da coluna {column}: {e}")

        return suggestions


# Instância singleton
_csv_processor: Optional[CSVProcessor] = None


def get_csv_processor() -> CSVProcessor:
    """Obtém instância singleton do CSVProcessor."""
    global _csv_processor
    if _csv_processor is None:
        _csv_processor = CSVProcessor()
    return _csv_processor