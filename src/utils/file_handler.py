"""
Manipulador robusto de arquivos CSV com deteccao automatica de encoding,
validacao de estrutura e integracao com o sistema de configuracao.
"""

import pandas as pd
import io
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import mimetypes
import tempfile
import shutil

from ..core.config import get_config
from ..core.logger import get_logger, log_data_operation, log_error_with_context, log_security_event


class FileHandlerError(Exception):
    """Erro especifico do manipulador de arquivos."""
    pass


class CSVFileHandler:
    """Manipulador robusto para arquivos CSV."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("file_handler")
        self.allowed_extensions = self.config.get_allowed_extensions()
        self.max_file_size = self.config.max_file_size_mb * 1024 * 1024  # Converter para bytes

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Valida se o arquivo atende aos criterios de seguranca e formato.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            True se o arquivo e valido

        Raises:
            FileHandlerError: Se o arquivo nao atender aos criterios
        """
        try:
            file_path = Path(file_path)

            # Verificar se o arquivo existe
            if not file_path.exists():
                raise FileHandlerError(f"Arquivo nao encontrado: {file_path}")

            # Verificar extensao
            extension = file_path.suffix.lower()
            if extension not in self.allowed_extensions:
                log_security_event(
                    self.logger,
                    "invalid_file_extension",
                    {"extension": extension, "file": str(file_path)}
                )
                raise FileHandlerError(f"Extensao nao permitida: {extension}")

            # Verificar tamanho
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                log_security_event(
                    self.logger,
                    "file_size_exceeded",
                    {"size_mb": file_size / (1024 * 1024), "max_mb": self.config.max_file_size_mb}
                )
                raise FileHandlerError(f"Arquivo muito grande: {file_size / (1024 * 1024):.1f}MB")

            # Verificar MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            allowed_mimes = ['text/csv', 'text/plain', 'application/csv']
            if mime_type and mime_type not in allowed_mimes:
                log_security_event(
                    self.logger,
                    "invalid_mime_type",
                    {"mime_type": mime_type, "file": str(file_path)}
                )
                raise FileHandlerError(f"Tipo de arquivo nao permitido: {mime_type}")

            log_data_operation(
                self.logger,
                "file_validation_success",
                {"file": str(file_path), "size_mb": file_size / (1024 * 1024)}
            )

            return True

        except Exception as e:
            log_error_with_context(
                self.logger,
                "file_validation_error",
                e,
                {"file_path": str(file_path)}
            )
            raise FileHandlerError(f"Erro na validacao do arquivo: {str(e)}")

    def detect_encoding(self, file_path: Union[str, Path]) -> str:
        """
        Detecta automaticamente o encoding do arquivo.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            String com o encoding detectado
        """
        try:
            file_path = Path(file_path)

            # Ler uma amostra do arquivo para deteccao
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Ler primeiros 10KB

            # Detectar encoding
            if HAS_CHARDET:
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0.0)

                # Se a confianca for muito baixa, usar utf-8 como fallback
                if confidence < 0.7:
                    encoding = 'utf-8'
                    self.logger.warning(f"Baixa confianca na deteccao de encoding ({confidence:.2f}), usando utf-8")
            else:
                # Fallback sem chardet - tentar detectar baseado em BOM ou usar utf-8
                if raw_data.startswith(b'\xff\xfe'):
                    encoding = 'utf-16-le'
                elif raw_data.startswith(b'\xfe\xff'):
                    encoding = 'utf-16-be'
                elif raw_data.startswith(b'\xef\xbb\xbf'):
                    encoding = 'utf-8-sig'
                else:
                    encoding = 'utf-8'
                confidence = 0.8

            log_data_operation(
                self.logger,
                "encoding_detection",
                {"file": str(file_path), "encoding": encoding, "confidence": confidence}
            )

            return encoding

        except Exception as e:
            log_error_with_context(
                self.logger,
                "encoding_detection_error",
                e,
                {"file_path": str(file_path)}
            )
            return 'utf-8'  # Fallback seguro

    def detect_csv_parameters(self, file_path: Union[str, Path], encoding: str) -> Dict[str, Any]:
        """
        Detecta automaticamente parametros do CSV (separador, aspas, etc.).

        Args:
            file_path: Caminho para o arquivo
            encoding: Encoding do arquivo

        Returns:
            Dicionario com parametros detectados
        """
        try:
            file_path = Path(file_path)

            # Ler amostra do arquivo
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(8192)  # Ler primeiros 8KB

            # Usar o Sniffer do pandas para detectar parametros
            sniffer = pd.io.common.CParserError

            # Tentar diferentes separadores
            separators = [',', ';', '\t', '|']
            best_params = {'sep': ',', 'quotechar': '"', 'skipinitialspace': True}

            for sep in separators:
                try:
                    # Testar leitura com este separador
                    test_df = pd.read_csv(
                        io.StringIO(sample),
                        sep=sep,
                        nrows=10,
                        skipinitialspace=True
                    )

                    # Verificar se o resultado parece bom
                    if test_df.shape[1] > 1 and test_df.shape[0] > 0:
                        best_params['sep'] = sep
                        break

                except Exception:
                    continue

            log_data_operation(
                self.logger,
                "csv_parameters_detection",
                {"file": str(file_path), "parameters": best_params}
            )

            return best_params

        except Exception as e:
            log_error_with_context(
                self.logger,
                "csv_parameters_detection_error",
                e,
                {"file_path": str(file_path)}
            )
            # Retornar parametros padrao
            return {'sep': ',', 'quotechar': '"', 'skipinitialspace': True}

    def read_csv_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Le arquivo CSV com deteccao automatica de parametros.

        Args:
            file_path: Caminho para o arquivo
            **kwargs: Parametros adicionais para pandas.read_csv

        Returns:
            DataFrame com os dados carregados

        Raises:
            FileHandlerError: Se houver erro na leitura
        """
        try:
            file_path = Path(file_path)

            # Validar arquivo
            self.validate_file(file_path)

            # Detectar encoding
            encoding = self.detect_encoding(file_path)

            # Detectar parametros CSV
            csv_params = self.detect_csv_parameters(file_path, encoding)

            # Mesclar parametros detectados com os fornecidos
            final_params = {**csv_params, **kwargs}
            final_params['encoding'] = encoding

            # Ler o arquivo
            df = pd.read_csv(file_path, **final_params)

            # Validacao basica dos dados
            if df.empty:
                raise FileHandlerError("Arquivo CSV esta vazio")

            if df.shape[1] == 0:
                raise FileHandlerError("Nenhuma coluna detectada no CSV")

            log_data_operation(
                self.logger,
                "csv_read_success",
                {
                    "file": str(file_path),
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "encoding": encoding,
                    "parameters": csv_params
                }
            )

            return df

        except pd.errors.EmptyDataError:
            raise FileHandlerError("Arquivo CSV esta vazio ou sem dados validos")
        except pd.errors.ParserError as e:
            raise FileHandlerError(f"Erro ao analisar CSV: {str(e)}")
        except UnicodeDecodeError as e:
            raise FileHandlerError(f"Erro de encoding: {str(e)}")
        except Exception as e:
            log_error_with_context(
                self.logger,
                "csv_read_error",
                e,
                {"file_path": str(file_path)}
            )
            raise FileHandlerError(f"Erro ao ler arquivo CSV: {str(e)}")

    def validate_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida a estrutura do DataFrame carregado.

        Args:
            df: DataFrame para validar

        Returns:
            Dicionario com informacoes de validacao
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

            # Calcular uso de memoria
            memory_usage = df.memory_usage(deep=True).sum()
            validation_info["memory_usage_mb"] = memory_usage / (1024 * 1024)

            # Verificar se o DataFrame e muito grande
            if validation_info["memory_usage_mb"] > 500:  # 500MB
                validation_info["warnings"].append(
                    f"Dataset grande ({validation_info['memory_usage_mb']:.1f}MB), "
                    "operacoes podem ser lentas"
                )

            log_data_operation(
                self.logger,
                "csv_structure_validation",
                validation_info
            )

            return validation_info

        except Exception as e:
            log_error_with_context(
                self.logger,
                "csv_structure_validation_error",
                e,
                {"dataframe_shape": df.shape if 'df' in locals() else "unknown"}
            )
            return {
                "is_valid": False,
                "error": str(e),
                "warnings": [f"Erro na validacao da estrutura: {str(e)}"]
            }

    def create_sample_dataset(self, df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
        """
        Cria uma amostra do dataset para analises rapidas.

        Args:
            df: DataFrame original
            sample_size: Tamanho da amostra

        Returns:
            DataFrame com amostra dos dados
        """
        try:
            if len(df) <= sample_size:
                return df.copy()

            # Criar amostra estratificada se possivel
            sample_df = df.sample(n=sample_size, random_state=42).copy()

            log_data_operation(
                self.logger,
                "sample_creation",
                {
                    "original_shape": df.shape,
                    "sample_shape": sample_df.shape,
                    "sample_ratio": sample_size / len(df)
                }
            )

            return sample_df

        except Exception as e:
            log_error_with_context(
                self.logger,
                "sample_creation_error",
                e,
                {"original_shape": df.shape, "sample_size": sample_size}
            )
            # Retornar as primeiras linhas como fallback
            return df.head(sample_size).copy()


# Instancia singleton
_file_handler: Optional[CSVFileHandler] = None


def get_file_handler() -> CSVFileHandler:
    """Obtem instancia singleton do manipulador de arquivos."""
    global _file_handler
    if _file_handler is None:
        _file_handler = CSVFileHandler()
    return _file_handler


# Funcoes de conveniencia
def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Funcao de conveniencia para leitura de CSV."""
    return get_file_handler().read_csv_file(file_path, **kwargs)


def validate_csv_file(file_path: Union[str, Path]) -> bool:
    """Funcao de conveniencia para validacao de arquivo."""
    return get_file_handler().validate_file(file_path)


def get_csv_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Funcao de conveniencia para informacoes da estrutura."""
    return get_file_handler().validate_csv_structure(df)