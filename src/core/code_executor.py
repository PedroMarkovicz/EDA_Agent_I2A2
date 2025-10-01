"""
Executor de código Python com sandboxing e medidas de segurança.
Executa código gerado pelos agentes de análise de forma controlada,
com timeout configurável, restrições de imports e captura segura
de outputs e erros para o sistema EDA.
"""

import ast
import sys
import subprocess
import tempfile
import signal
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import traceback
import io
import contextlib
from dataclasses import dataclass

from .config import get_config
from .logger import get_logger, log_data_operation, log_error_with_context, log_security_event


@dataclass
class ExecutionResult:
    """Resultado da execução de código."""
    success: bool
    output: str
    error: str
    execution_time: float
    variables_captured: Dict[str, Any]
    plots_generated: List[str]
    metadata: Dict[str, Any]


class CodeSecurityError(Exception):
    """Erro de segurança na execução de código."""
    pass


class CodeExecutionTimeout(Exception):
    """Timeout na execução de código."""
    pass


class CodeExecutor:
    """Executor seguro de código Python."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("code_executor")

        # Configurações de segurança
        self.safe_mode = self.config.safe_mode
        self.execution_timeout = self.config.execution_timeout

        # Imports permitidos no modo seguro
        self.allowed_imports = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'scipy', 'sklearn', 'math', 'statistics', 'datetime',
            'json', 'csv', 're', 'collections', 'itertools',
            'warnings', 'typing'
        }

        # Imports proibidos (sempre)
        self.forbidden_imports = {
            'os', 'sys', 'subprocess', 'socket', 'urllib',
            'requests', 'http', 'ftplib', 'smtplib', 'telnetlib',
            'webbrowser', 'pickle', 'marshal', 'shelve',
            '__import__', 'eval', 'exec', 'compile', 'open',
            'file', 'input', 'raw_input'
        }

        # Diretório para plots temporários
        self.temp_plots_dir = Path(tempfile.gettempdir()) / "eda_agent_plots"
        self.temp_plots_dir.mkdir(exist_ok=True)

    def validate_code_security(self, code: str) -> Tuple[bool, List[str]]:
        """
        Valida segurança do código antes da execução.

        Args:
            code: Código Python a ser validado

        Returns:
            Tuple (é_seguro, lista_de_problemas)
        """
        problems = []

        try:
            # Parse do AST para análise estática
            tree = ast.parse(code)

            # Visitor para detectar construções perigosas
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []

                def visit_Import(self, node):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name in self.forbidden_imports:
                            self.issues.append(f"Import proibido: {module_name}")
                        elif self.safe_mode and module_name not in self.allowed_imports:
                            self.issues.append(f"Import não permitido em modo seguro: {module_name}")

                def visit_ImportFrom(self, node):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name in self.forbidden_imports:
                            self.issues.append(f"Import proibido: {module_name}")
                        elif self.safe_mode and module_name not in self.allowed_imports:
                            self.issues.append(f"Import não permitido em modo seguro: {module_name}")

                def visit_Call(self, node):
                    # Detectar chamadas perigosas
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        dangerous_functions = ['exec', 'eval', 'compile', '__import__', 'open', 'input']
                        if func_name in dangerous_functions:
                            self.issues.append(f"Função perigosa: {func_name}")

                    # Detectar tentativas de acesso a atributos do sistema
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            if node.func.value.id in ['os', 'sys', 'subprocess']:
                                self.issues.append(f"Acesso não permitido: {node.func.value.id}.{node.func.attr}")

                    self.generic_visit(node)

                def visit_Attribute(self, node):
                    # Detectar acesso a atributos privados/perigosos
                    if node.attr.startswith('_') and len(node.attr) > 1:
                        self.issues.append(f"Acesso a atributo privado: {node.attr}")
                    self.generic_visit(node)

            visitor = SecurityVisitor()
            visitor.visit(tree)
            problems.extend(visitor.issues)

            # Verificar strings suspeitas no código
            suspicious_patterns = [
                'rm -rf', 'del ', 'remove(', 'unlink(',
                'system(', 'popen(', 'spawn(',
                'http://', 'https://', 'ftp://',
                'socket.', 'urllib.', 'requests.',
                '__builtins__', '__globals__', '__locals__'
            ]

            code_lower = code.lower()
            for pattern in suspicious_patterns:
                if pattern in code_lower:
                    problems.append(f"Padrão suspeito detectado: {pattern}")

        except SyntaxError as e:
            problems.append(f"Erro de sintaxe: {e}")

        except Exception as e:
            problems.append(f"Erro na validação: {e}")

        is_safe = len(problems) == 0
        return is_safe, problems

    def _setup_execution_environment(self, code: str) -> str:
        """Prepara ambiente de execução com configurações de segurança."""
        setup_code = """
import sys
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para não usar interface gráfica
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configurar pandas para display
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 120)

# Variáveis para captura
_captured_vars = {}
_plot_files = []

# Função para salvar plots
def save_plot(filename=None):
    import uuid
    if filename is None:
        filename = f"plot_{uuid.uuid4().hex[:8]}.png"

    full_path = f"{str(temp_plots_dir)}/{filename}"
    plt.savefig(full_path, dpi=100, bbox_inches='tight')
    _plot_files.append(full_path)
    return full_path

# Sobrescrever plt.show para salvar plots automaticamente
original_show = plt.show
def custom_show():
    save_plot()
    plt.close()
plt.show = custom_show

"""

        # Injetar variáveis necessárias
        setup_code = setup_code.replace("temp_plots_dir", f'"{self.temp_plots_dir}"')

        return setup_code + "\n" + code

    def execute_code(
        self,
        code: str,
        context_vars: Optional[Dict[str, Any]] = None,
        capture_output: bool = True
    ) -> ExecutionResult:
        """
        Executa código Python de forma segura.

        Args:
            code: Código Python para executar
            context_vars: Variáveis de contexto (ex: DataFrame)
            capture_output: Se deve capturar stdout/stderr

        Returns:
            ExecutionResult com resultado da execução
        """
        start_time = datetime.now()

        try:
            # Validar segurança do código
            is_safe, security_issues = self.validate_code_security(code)

            if not is_safe:
                log_security_event(
                    "code_security_violation",
                    {"issues": security_issues, "code_snippet": code[:200]}
                )
                raise CodeSecurityError(f"Código inseguro: {'; '.join(security_issues)}")

            # Preparar código para execução
            full_code = self._setup_execution_environment(code)

            # Preparar namespace de execução
            exec_namespace = {
                '__builtins__': self._create_safe_builtins(),
                'temp_plots_dir': self.temp_plots_dir
            }

            # Adicionar variáveis de contexto
            if context_vars:
                exec_namespace.update(context_vars)

            # Capturar output se solicitado
            if capture_output:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()

                try:
                    with contextlib.redirect_stdout(stdout_capture), \
                         contextlib.redirect_stderr(stderr_capture):

                        # Executar com timeout
                        self._execute_with_timeout(full_code, exec_namespace)

                    output = stdout_capture.getvalue()
                    error = stderr_capture.getvalue()

                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            else:
                # Executar sem captura
                self._execute_with_timeout(full_code, exec_namespace)
                output = ""
                error = ""

            # Capturar variáveis geradas
            captured_vars = {}
            for key, value in exec_namespace.items():
                if not key.startswith('_') and key not in ['temp_plots_dir']:
                    try:
                        # Tentar serializar para verificar se é capturável
                        str(value)
                        captured_vars[key] = value
                    except Exception:
                        captured_vars[key] = f"<{type(value).__name__}>"

            # Obter plots gerados
            plots_generated = exec_namespace.get('_plot_files', [])

            execution_time = (datetime.now() - start_time).total_seconds()

            result = ExecutionResult(
                success=True,
                output=output,
                error=error,
                execution_time=execution_time,
                variables_captured=captured_vars,
                plots_generated=plots_generated,
                metadata={
                    "code_length": len(code),
                    "namespace_size": len(exec_namespace),
                    "timestamp": start_time.isoformat()
                }
            )

            log_data_operation(
                "code_executed",
                {
                    "success": True,
                    "execution_time": execution_time,
                    "output_length": len(output),
                    "plots_count": len(plots_generated)
                }
            )

            return result

        except CodeExecutionTimeout:
            error_msg = f"Execução excedeu tempo limite de {self.execution_timeout}s"
            log_error_with_context(
                CodeExecutionTimeout(error_msg),
                {"timeout": self.execution_timeout, "code_length": len(code)}
            )

            return ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                execution_time=self.execution_timeout,
                variables_captured={},
                plots_generated=[],
                metadata={"timeout": True}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erro na execução: {str(e)}\n{traceback.format_exc()}"

            log_error_with_context(
                e,
                {
                    "code_length": len(code),
                    "execution_time": execution_time,
                    "error_type": type(e).__name__
                }
            )

            return ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                execution_time=execution_time,
                variables_captured={},
                plots_generated=[],
                metadata={"error_type": type(e).__name__}
            )

    def _execute_with_timeout(self, code: str, namespace: Dict[str, Any]) -> None:
        """Executa código com timeout."""
        if os.name == 'nt':  # Windows
            # No Windows, usar subprocess para timeout
            self._execute_with_subprocess_timeout(code, namespace)
        else:  # Unix-like systems
            # Usar signal para timeout
            def timeout_handler(signum, frame):
                raise CodeExecutionTimeout("Timeout durante execução")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.execution_timeout)

            try:
                exec(code, namespace)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def _execute_with_subprocess_timeout(self, code: str, namespace: Dict[str, Any]) -> None:
        """Executa código usando subprocess com timeout (para Windows)."""
        # Para Windows, executar diretamente com exec
        # Nota: Timeout mais limitado no Windows
        import threading

        result = {"exception": None}

        def target():
            try:
                exec(code, namespace)
            except Exception as e:
                result["exception"] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.execution_timeout)

        if thread.is_alive():
            # Timeout - tentar forçar parada (limitado no Windows)
            raise CodeExecutionTimeout("Timeout durante execução")

        if result["exception"]:
            raise result["exception"]

    def _create_safe_builtins(self) -> Dict[str, Any]:
        """Cria versão segura de builtins."""
        safe_builtins = {
            # Funções matemáticas e utilitárias seguras
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
            'chr', 'complex', 'dict', 'enumerate', 'filter', 'float',
            'frozenset', 'hex', 'int', 'len', 'list', 'map', 'max',
            'min', 'oct', 'ord', 'pow', 'range', 'reversed', 'round',
            'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type',
            'zip',

            # Tipos de exceção
            'Exception', 'ValueError', 'TypeError', 'IndexError',
            'KeyError', 'AttributeError',

            # Constantes
            'True', 'False', 'None',

            # Funções de análise seguras
            'print', 'isinstance', 'hasattr', 'getattr', 'setattr',
            'dir', 'vars', 'help'
        }

        # Construir dicionário com builtins seguros
        import builtins
        safe_builtins_dict = {}

        for name in safe_builtins:
            if hasattr(builtins, name):
                safe_builtins_dict[name] = getattr(builtins, name)

        return safe_builtins_dict

    def cleanup_temp_files(self) -> int:
        """
        Remove arquivos temporários de plots antigos.

        Returns:
            Número de arquivos removidos
        """
        try:
            removed_count = 0
            cutoff_time = datetime.now().timestamp() - (3600 * 24)  # 24 horas

            for plot_file in self.temp_plots_dir.glob("plot_*.png"):
                try:
                    if plot_file.stat().st_mtime < cutoff_time:
                        plot_file.unlink()
                        removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Erro ao remover {plot_file}: {e}")

            log_data_operation(
                "temp_files_cleaned",
                {"removed_count": removed_count}
            )

            return removed_count

        except Exception as e:
            log_error_with_context(e, {"operation": "cleanup_temp_files"})
            return 0

    def get_execution_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do executor."""
        try:
            plot_files = list(self.temp_plots_dir.glob("plot_*.png"))
            total_size = sum(f.stat().st_size for f in plot_files) / (1024 * 1024)  # MB

            return {
                "safe_mode": self.safe_mode,
                "execution_timeout": self.execution_timeout,
                "temp_plots_count": len(plot_files),
                "temp_plots_size_mb": round(total_size, 2),
                "temp_plots_dir": str(self.temp_plots_dir),
                "allowed_imports_count": len(self.allowed_imports),
                "forbidden_imports_count": len(self.forbidden_imports)
            }

        except Exception as e:
            log_error_with_context(e, {"operation": "get_execution_stats"})
            return {"error": str(e)}


# Instância singleton
_code_executor: Optional[CodeExecutor] = None


def get_code_executor() -> CodeExecutor:
    """Obtém instância singleton do CodeExecutor."""
    global _code_executor
    if _code_executor is None:
        _code_executor = CodeExecutor()
    return _code_executor