"""
Registro central de ferramentas estatísticas.
Gerencia descoberta, registro e execução de ferramentas.
"""

from typing import Dict, List, Optional, Type, Any
import pandas as pd
from datetime import datetime

from .base import BaseTool, ToolResult
from ..core.logger import get_logger


class ToolRegistry:
    """
    Registro central de todas as ferramentas estatísticas disponíveis.

    Responsável por:
    - Registrar ferramentas
    - Fornecer descrições para o LLM
    - Executar ferramentas com validação
    - Gerenciar cache de resultados
    """

    def __init__(self):
        self.logger = get_logger("tool_registry")
        self._tools: Dict[str, BaseTool] = {}
        self._cache: Dict[str, ToolResult] = {}
        self._execution_history: List[Dict[str, Any]] = []

    def register_tool(self, tool_instance: BaseTool) -> None:
        """
        Registra uma nova ferramenta no registry.

        Args:
            tool_instance: Instância da ferramenta a registrar
        """
        tool_name = tool_instance.name

        if tool_name in self._tools:
            self.logger.warning(f"Ferramenta '{tool_name}' já registrada. Sobrescrevendo.")

        self._tools[tool_name] = tool_instance
        self.logger.info(f"Ferramenta registrada: {tool_name} (categoria: {tool_instance.category})")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Obtém uma ferramenta pelo nome.

        Args:
            tool_name: Nome da ferramenta

        Returns:
            Instância da ferramenta ou None se não encontrada
        """
        return self._tools.get(tool_name)

    def get_all_tools(self) -> Dict[str, BaseTool]:
        """
        Retorna todas as ferramentas registradas.

        Returns:
            Dicionário {nome: ferramenta}
        """
        return self._tools.copy()

    def get_tools_by_category(self, category: str) -> Dict[str, BaseTool]:
        """
        Retorna ferramentas de uma categoria específica.

        Args:
            category: Categoria das ferramentas

        Returns:
            Dicionário com ferramentas da categoria
        """
        return {
            name: tool
            for name, tool in self._tools.items()
            if tool.category == category
        }

    def get_all_tools_description(self) -> List[Dict[str, Any]]:
        """
        Retorna descrição de todas as ferramentas para o LLM.

        Returns:
            Lista de descrições estruturadas
        """
        return [
            tool.get_description()
            for tool in self._tools.values()
        ]

    def get_tools_description_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retorna descrições agrupadas por categoria.

        Returns:
            Dicionário {categoria: [descrições]}
        """
        descriptions_by_category: Dict[str, List[Dict[str, Any]]] = {}

        for tool in self._tools.values():
            category = tool.category
            if category not in descriptions_by_category:
                descriptions_by_category[category] = []

            descriptions_by_category[category].append(tool.get_description())

        return descriptions_by_category

    def execute_tool(
        self,
        tool_name: str,
        df: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> ToolResult:
        """
        Executa uma ferramenta com os parâmetros fornecidos.

        Args:
            tool_name: Nome da ferramenta a executar
            df: DataFrame com os dados
            parameters: Parâmetros para a ferramenta
            use_cache: Se deve usar cache de resultados

        Returns:
            ToolResult com resultado da execução
        """
        start_time = datetime.now()
        parameters = parameters or {}

        # Verificar se ferramenta existe
        tool = self.get_tool(tool_name)
        if not tool:
            error_msg = f"Ferramenta '{tool_name}' não encontrada"
            self.logger.error(error_msg)
            return ToolResult(
                success=False,
                data={},
                error=error_msg
            )

        # Gerar chave de cache
        cache_key = self._generate_cache_key(tool_name, df, parameters)

        # Verificar cache
        if use_cache and cache_key in self._cache:
            self.logger.info(f"Resultado obtido do cache para: {tool_name}")
            cached_result = self._cache[cache_key]
            cached_result.metadata['from_cache'] = True
            return cached_result

        # Validar parâmetros
        is_valid, error_msg = tool.validate_parameters(**parameters)
        if not is_valid:
            self.logger.error(f"Parâmetros inválidos para {tool_name}: {error_msg}")
            return ToolResult(
                success=False,
                data={},
                error=error_msg
            )

        # Executar ferramenta
        try:
            self.logger.info(f"Executando ferramenta: {tool_name} com parâmetros: {parameters}")
            result = tool.execute(df, **parameters)

            # Calcular tempo de execução
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            # Adicionar ao cache se bem-sucedido
            if result.success and use_cache:
                self._cache[cache_key] = result

            # Adicionar ao histórico
            self._execution_history.append({
                'tool_name': tool_name,
                'parameters': parameters,
                'success': result.success,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            })

            self.logger.info(
                f"Ferramenta {tool_name} executada com sucesso "
                f"(tempo: {execution_time:.3f}s)"
            )

            return result

        except Exception as e:
            error_msg = f"Erro ao executar {tool_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return ToolResult(
                success=False,
                data={},
                error=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _generate_cache_key(
        self,
        tool_name: str,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Gera chave única para cache.

        Args:
            tool_name: Nome da ferramenta
            df: DataFrame
            parameters: Parâmetros da ferramenta

        Returns:
            Chave de cache
        """
        import hashlib
        import json

        # Gerar hash do DataFrame (baseado em shape e colunas)
        df_signature = f"{df.shape}_{list(df.columns)}"

        # Gerar hash dos parâmetros
        params_str = json.dumps(parameters, sort_keys=True)

        # Combinar tudo
        cache_string = f"{tool_name}_{df_signature}_{params_str}"

        # Gerar hash MD5
        return hashlib.md5(cache_string.encode()).hexdigest()

    def clear_cache(self) -> int:
        """
        Limpa o cache de resultados.

        Returns:
            Número de itens removidos
        """
        count = len(self._cache)
        self._cache = {}
        self.logger.info(f"Cache limpo: {count} itens removidos")
        return count

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retorna histórico de execuções.

        Args:
            limit: Número máximo de itens a retornar

        Returns:
            Lista com histórico de execuções
        """
        if limit:
            return self._execution_history[-limit:]
        return self._execution_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre o uso das ferramentas.

        Returns:
            Dicionário com estatísticas
        """
        total_executions = len(self._execution_history)
        successful_executions = sum(
            1 for item in self._execution_history if item['success']
        )

        # Ferramentas mais usadas
        tool_usage = {}
        for item in self._execution_history:
            tool_name = item['tool_name']
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        # Tempo médio por ferramenta
        avg_time_by_tool = {}
        for tool_name in tool_usage:
            times = [
                item['execution_time']
                for item in self._execution_history
                if item['tool_name'] == tool_name
            ]
            avg_time_by_tool[tool_name] = sum(times) / len(times) if times else 0

        return {
            'total_tools_registered': len(self._tools),
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': total_executions - successful_executions,
            'cache_size': len(self._cache),
            'tool_usage': tool_usage,
            'avg_execution_time_by_tool': avg_time_by_tool
        }


# Instância global do registry
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """
    Obtém instância singleton do ToolRegistry.

    Returns:
        Instância do ToolRegistry
    """
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        _initialize_default_tools(_tool_registry)
    return _tool_registry


def _initialize_default_tools(registry: ToolRegistry) -> None:
    """
    Inicializa e registra ferramentas padrão.

    Args:
        registry: Instância do registry
    """
    from .basic_stats import BasicStatsTool
    from .correlation_analysis import CorrelationTool
    from .outlier_detection import OutlierDetectionTool
    from .missing_data_analysis import MissingDataTool
    from .schema_analysis import SchemaAnalysisTool
    from .visualization_tools import (
        PlotDistributionTool,
        PlotBoxplotTool,
        PlotCorrelationHeatmapTool,
        PlotBarChartTool
    )
    from .minimal_stats import (
        GetMeanTool,
        GetMedianTool,
        GetMaxTool,
        GetMinTool,
        GetSumTool,
        GetStdTool,
        GetCountTool
    )

    # Registrar ferramenta de schema primeiro (útil para queries conceituais)
    registry.register_tool(SchemaAnalysisTool())

    # Registrar ferramentas de visualização (alta prioridade para queries visuais)
    registry.register_tool(PlotDistributionTool())
    registry.register_tool(PlotBoxplotTool())
    registry.register_tool(PlotCorrelationHeatmapTool())
    registry.register_tool(PlotBarChartTool())

    # Registrar ferramentas minimalistas (prioridade maior)
    registry.register_tool(GetMeanTool())
    registry.register_tool(GetMedianTool())
    registry.register_tool(GetMaxTool())
    registry.register_tool(GetMinTool())
    registry.register_tool(GetSumTool())
    registry.register_tool(GetStdTool())
    registry.register_tool(GetCountTool())

    # Registrar ferramentas abrangentes
    registry.register_tool(BasicStatsTool())
    registry.register_tool(CorrelationTool())
    registry.register_tool(OutlierDetectionTool())
    registry.register_tool(MissingDataTool())
