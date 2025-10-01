"""
Funcoes de seguranca para validacao de entrada e execucao segura de codigo.
Implementa sandbox para codigo Python e verificacoes de seguranca.
"""

import ast
import re
import sys
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import importlib.util
import builtins

from ..core.config import get_config
from ..core.logger import get_logger, log_security_event, log_error_with_context


class SecurityError(Exception):
    """Erro de seguranca."""
    pass


class CodeSecurityValidator:
    """Validador de seguranca para codigo Python."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("security")
        self.restricted_imports = set(self.config.restricted_imports)
        self.dangerous_builtins = {
            'eval', 'exec', 'compile', '__import__', 'open', 'input',
            'raw_input', 'file', 'execfile', 'reload', 'vars', 'locals',
            'globals', 'dir', 'help', 'memoryview', 'breakpoint'
        }

    def validate_python_code(self, code: str, context: str = "general") -> Dict[str, Any]:
        """
        Valida codigo Python para execucao segura.

        Args:
            code: Codigo Python a ser validado
            context: Contexto da validacao

        Returns:
            Resultado da validacao com detalhes de seguranca
        """
        try:
            validation_result = {
                "is_safe": True,
                "security_issues": [],
                "warnings": [],
                "blocked_operations": [],
                "allowed_imports": [],
                "code_metrics": {
                    "lines": 0,
                    "complexity": 0,
                    "ast_nodes": 0
                }
            }

            if not code.strip():
                validation_result["security_issues"].append("Codigo vazio")
                validation_result["is_safe"] = False
                return validation_result

            # Analise estatica do codigo
            try:
                tree = ast.parse(code)
                validation_result["code_metrics"]["ast_nodes"] = len(list(ast.walk(tree)))
                validation_result["code_metrics"]["lines"] = len(code.splitlines())
            except SyntaxError as e:
                validation_result["security_issues"].append(f"Erro de sintaxe: {str(e)}")
                validation_result["is_safe"] = False
                return validation_result

            # Verificar imports perigosos
            self._check_dangerous_imports(tree, validation_result)

            # Verificar builtins perigosos
            self._check_dangerous_builtins(tree, validation_result)

            # Verificar operacoes de arquivo
            self._check_file_operations(tree, validation_result)

            # Verificar chamadas de sistema
            self._check_system_calls(tree, validation_result)

            # Verificar manipulacao de atributos dinamicos
            self._check_dynamic_attribute_access(tree, validation_result)

            # Verificar padroes de codigo suspeitos
            self._check_suspicious_patterns(code, validation_result)

            # Se houveram problemas de seguranca, marcar como inseguro
            if validation_result["security_issues"]:
                validation_result["is_safe"] = False

            log_security_event(
                self.logger,
                "code_validation_completed",
                {
                    "context": context,
                    "is_safe": validation_result["is_safe"],
                    "issues_count": len(validation_result["security_issues"]),
                    "code_lines": validation_result["code_metrics"]["lines"]
                }
            )

            return validation_result

        except Exception as e:
            log_error_with_context(
                e,
                {"context": context, "code_preview": code[:100]},
                "code_validation_error"
            )
            return {
                "is_safe": False,
                "security_issues": [f"Erro na validacao de seguranca: {str(e)}"],
                "warnings": [],
                "blocked_operations": [],
                "allowed_imports": [],
                "code_metrics": {}
            }

    def _check_dangerous_imports(self, tree: ast.AST, result: Dict[str, Any]):
        """Verifica imports perigosos no codigo."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    if self._is_import_restricted(module_name):
                        result["security_issues"].append(f"Import restrito: {module_name}")
                        result["blocked_operations"].append(f"import {module_name}")
                    else:
                        result["allowed_imports"].append(module_name)

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if self._is_import_restricted(module_name):
                    result["security_issues"].append(f"Import restrito: {module_name}")
                    result["blocked_operations"].append(f"from {module_name} import ...")
                else:
                    result["allowed_imports"].append(module_name)

    def _check_dangerous_builtins(self, tree: ast.AST, result: Dict[str, Any]):
        """Verifica uso de builtins perigosos."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.dangerous_builtins:
                        result["security_issues"].append(f"Funcao perigosa: {func_name}")
                        result["blocked_operations"].append(func_name)

    def _check_file_operations(self, tree: ast.AST, result: Dict[str, Any]):
        """Verifica operacoes de arquivo potencialmente perigosas."""
        file_functions = {'open', 'file', 'read', 'write', 'remove', 'unlink', 'rmdir'}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in file_functions:
                        result["warnings"].append(f"Operacao de arquivo detectada: {func_name}")

    def _check_system_calls(self, tree: ast.AST, result: Dict[str, Any]):
        """Verifica chamadas de sistema."""
        system_modules = {'subprocess', 'os', 'sys', 'shutil', 'tempfile'}

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in system_modules:
                        result["warnings"].append(
                            f"Acesso a modulo de sistema: {node.value.id}.{node.attr}"
                        )

    def _check_dynamic_attribute_access(self, tree: ast.AST, result: Dict[str, Any]):
        """Verifica acesso dinamico a atributos."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['getattr', 'setattr', 'delattr', 'hasattr']:
                        result["warnings"].append(f"Acesso dinamico a atributos: {func_name}")

    def _check_suspicious_patterns(self, code: str, result: Dict[str, Any]):
        """Verifica padroes suspeitos no codigo."""
        suspicious_patterns = [
            (r'__\w+__', "Uso de metodos magicos"),
            (r'\.format\s*\(.*\{.*\}.*\)', "String formatting potencialmente perigoso"),
            (r'%\s*\(.*\)', "String interpolation"),
            (r'pickle\.loads?', "Deserializacao perigosa com pickle"),
            (r'marshal\.loads?', "Deserializacao com marshal"),
            (r'subprocess\.', "Execucao de processo"),
            (r'os\.system', "Execucao de comando do sistema"),
            (r'eval\s*\(', "Uso de eval"),
            (r'exec\s*\(', "Uso de exec")
        ]

        for pattern, description in suspicious_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                result["warnings"].append(description)

    def _is_import_restricted(self, module_name: str) -> bool:
        """Verifica se um import e restrito."""
        if not module_name:
            return False

        # Verificar imports diretamente restritos
        if module_name in self.restricted_imports:
            return True

        # Verificar prefixos de modulos restritos
        for restricted in self.restricted_imports:
            if module_name.startswith(f"{restricted}."):
                return True

        return False

    def create_safe_execution_environment(self) -> Dict[str, Any]:
        """
        Cria ambiente seguro para execucao de codigo Python.

        Returns:
            Dicionario com namespace seguro para execucao
        """
        try:
            # Builtins seguros permitidos
            safe_builtins = {
                'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
                'callable', 'chr', 'classmethod', 'complex', 'dict', 'divmod',
                'enumerate', 'filter', 'float', 'format', 'frozenset',
                'getattr', 'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance',
                'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min',
                'next', 'object', 'oct', 'ord', 'pow', 'property', 'range',
                'repr', 'reversed', 'round', 'set', 'setattr', 'slice',
                'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple',
                'type', 'zip', 'print'
            }

            # Criar namespace seguro
            safe_namespace = {
                '__builtins__': {name: getattr(builtins, name) for name in safe_builtins}
            }

            # Adicionar modulos seguros para EDA
            safe_modules = {
                'pandas': 'pd',
                'numpy': 'np',
                'matplotlib.pyplot': 'plt',
                'seaborn': 'sns',
                'scipy.stats': 'stats',
                'math': 'math',
                'statistics': 'statistics',
                'datetime': 'datetime'
            }

            for module_name, alias in safe_modules.items():
                try:
                    if importlib.util.find_spec(module_name.split('.')[0]):
                        exec(f"import {module_name} as {alias}", safe_namespace)
                except ImportError:
                    continue

            log_security_event(
                self.logger,
                "safe_environment_created",
                {"safe_builtins_count": len(safe_builtins), "safe_modules_count": len(safe_modules)}
            )

            return safe_namespace

        except Exception as e:
            log_error_with_context(
                e,
                {},
                "safe_environment_creation_error"
            )
            raise SecurityError(f"Erro ao criar ambiente seguro: {str(e)}")

    def execute_code_safely(self,
                           code: str,
                           local_vars: Optional[Dict[str, Any]] = None,
                           timeout: int = 30) -> Dict[str, Any]:
        """
        Executa codigo Python em ambiente seguro.

        Args:
            code: Codigo Python para executar
            local_vars: Variaveis locais para o contexto
            timeout: Timeout em segundos

        Returns:
            Resultado da execucao
        """
        try:
            # Validar codigo primeiro
            validation = self.validate_python_code(code, "execution")
            if not validation["is_safe"]:
                raise SecurityError(f"Codigo inseguro: {validation['security_issues']}")

            # Criar ambiente seguro
            safe_namespace = self.create_safe_execution_environment()

            # Adicionar variaveis locais se fornecidas
            if local_vars:
                safe_namespace.update(local_vars)

            # Capturar stdout
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            execution_result = {
                "success": True,
                "output": "",
                "error": None,
                "returned_value": None,
                "variables_created": []
            }

            try:
                # Executar codigo
                exec(code, safe_namespace)

                # Capturar variaveis criadas
                execution_result["variables_created"] = [
                    key for key in safe_namespace.keys()
                    if not key.startswith('__') and key not in ['pd', 'np', 'plt', 'sns', 'stats', 'math']
                ]

                execution_result["output"] = captured_output.getvalue()

            except Exception as e:
                execution_result["success"] = False
                execution_result["error"] = str(e)

            finally:
                sys.stdout = old_stdout

            log_security_event(
                self.logger,
                "code_executed_safely",
                {
                    "success": execution_result["success"],
                    "output_length": len(execution_result["output"]),
                    "variables_created": len(execution_result["variables_created"])
                }
            )

            return execution_result

        except Exception as e:
            log_error_with_context(
                e,
                {"code_preview": code[:100]},
                "safe_execution_error"
            )
            return {
                "success": False,
                "output": "",
                "error": f"Erro na execucao segura: {str(e)}",
                "returned_value": None,
                "variables_created": []
            }

    def sanitize_user_input(self, user_input: str) -> str:
        """
        Sanitiza entrada do usuario removendo conteudo perigoso.

        Args:
            user_input: Entrada do usuario

        Returns:
            Entrada sanitizada
        """
        try:
            # Remover caracteres de controle
            sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', user_input)

            # Remover tags HTML/XML
            sanitized = re.sub(r'<[^>]+>', '', sanitized)

            # Remover JavaScript
            sanitized = re.sub(r'javascript\s*:', '', sanitized, flags=re.IGNORECASE)

            # Remover data URIs
            sanitized = re.sub(r'data\s*:', '', sanitized, flags=re.IGNORECASE)

            # Limitar tamanho
            max_length = 1000
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length]

            return sanitized.strip()

        except Exception as e:
            log_error_with_context(
                e,
                {"input_preview": user_input[:100] if user_input else "empty"},
                "input_sanitization_error"
            )
            return ""

    def validate_file_path(self, file_path: str, allowed_extensions: Optional[Set[str]] = None) -> bool:
        """
        Valida caminho de arquivo para seguranca.

        Args:
            file_path: Caminho do arquivo
            allowed_extensions: Extensoes permitidas

        Returns:
            True se o caminho e seguro
        """
        try:
            path = Path(file_path)

            # Verificar se o caminho nao tenta escapar do diretorio
            if '..' in path.parts:
                log_security_event(
                    self.logger,
                    "path_traversal_attempt",
                    {"file_path": file_path}
                )
                return False

            # Verificar extensao se especificada
            if allowed_extensions and path.suffix.lower() not in allowed_extensions:
                return False

            # Verificar caracteres perigosos
            dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
            if any(char in file_path for char in dangerous_chars):
                return False

            return True

        except Exception as e:
            log_error_with_context(
                e,
                {"file_path": file_path},
                "file_path_validation_error"
            )
            return False


# Instancia singleton
_security_validator: Optional[CodeSecurityValidator] = None


def get_security_validator() -> CodeSecurityValidator:
    """Obtem instancia singleton do validador de seguranca."""
    global _security_validator
    if _security_validator is None:
        _security_validator = CodeSecurityValidator()
    return _security_validator


# Funcoes de conveniencia
def validate_python_code(code: str, context: str = "general") -> Dict[str, Any]:
    """Funcao de conveniencia para validacao de codigo."""
    return get_security_validator().validate_python_code(code, context)


def execute_code_safely(code: str, local_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Funcao de conveniencia para execucao segura."""
    return get_security_validator().execute_code_safely(code, local_vars)


def sanitize_user_input(user_input: str) -> str:
    """Funcao de conveniencia para sanitizacao de entrada."""
    return get_security_validator().sanitize_user_input(user_input)


def is_code_safe(code: str) -> bool:
    """Funcao simplificada para verificar se codigo e seguro."""
    result = validate_python_code(code)
    return result["is_safe"]


def is_file_path_safe(file_path: str, allowed_extensions: Optional[Set[str]] = None) -> bool:
    """Funcao simplificada para validacao de caminho de arquivo."""
    return get_security_validator().validate_file_path(file_path, allowed_extensions)