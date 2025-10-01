"""
Agente responsável pela geração dinâmica de código Python para análises EDA.
Cria scripts de análise específicos, códigos de visualização e implementa
análises personalizadas baseadas nas consultas dos usuários, sempre
focando em análise exploratória sem transformação de dados.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.config import get_config
from ..core.llm_manager import get_llm_manager
from ..core.code_executor import get_code_executor
from ..core.logger import get_logger, log_analysis_step, log_error_with_context
from ..models.enums import EDAAnalysisType, ProcessingStatus, VisualizationType


class CodeGenerationError(Exception):
    """Erro específico para geração de código."""
    pass


class CodeGeneratorAgent:
    """Agente especializado em geração de código Python para EDA."""

    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        self.code_executor = get_code_executor()
        self.logger = get_logger("code_generator_agent")

    def generate_analysis_code(
        self,
        query_text: str,
        analysis_types: List[EDAAnalysisType],
        context: Optional[Dict[str, Any]] = None,
        execute_code: bool = False
    ) -> Dict[str, Any]:
        """
        Gera código Python para análises EDA específicas.

        Args:
            query_text: Consulta do usuário
            analysis_types: Tipos de análise necessários
            context: Contexto adicional (metadados do CSV, etc.)
            execute_code: Se deve executar o código gerado

        Returns:
            Dict com código gerado e resultados da execução
        """
        start_time = time.time()

        try:
            log_analysis_step("code_generation", "started", {
                "query_length": len(query_text),
                "analysis_types": [t.value for t in analysis_types],
                "execute": execute_code
            })

            result = {
                "status": ProcessingStatus.IN_PROGRESS,
                "generated_code": "",
                "execution_result": None,
                "code_explanation": "",
                "visualizations_created": [],
                "errors": []
            }

            # 1. Gerar código baseado na consulta e tipos de análise
            generated_code = self._generate_code_for_analysis(query_text, analysis_types, context)
            result["generated_code"] = generated_code

            # 2. Gerar explicação do código
            code_explanation = self._generate_code_explanation(generated_code, query_text)
            result["code_explanation"] = code_explanation

            # 3. Executar código se solicitado
            if execute_code and generated_code:
                execution_result = self._execute_generated_code(generated_code, context)
                result["execution_result"] = execution_result

                # Coletar visualizações geradas
                if execution_result and execution_result.plots_generated:
                    result["visualizations_created"] = execution_result.plots_generated

            # 4. Marcar como concluído
            result["status"] = ProcessingStatus.COMPLETED
            result["processing_time"] = time.time() - start_time

            log_analysis_step(
                "code_generation", "completed",
                {
                    "processing_time": result["processing_time"],
                    "code_length": len(generated_code),
                    "executed": execute_code,
                    "visualizations": len(result["visualizations_created"])
                }
            )

            return result

        except Exception as e:
            log_error_with_context(e, {
                "operation": "code_generation",
                "query": query_text,
                "analysis_types": [t.value for t in analysis_types]
            })
            raise CodeGenerationError(f"Erro na geração de código: {e}")

    def generate_visualization_code(
        self,
        viz_type: VisualizationType,
        columns: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Gera código específico para visualizações.

        Args:
            viz_type: Tipo de visualização
            columns: Colunas a serem visualizadas
            context: Contexto adicional

        Returns:
            Código Python para a visualização
        """
        try:
            viz_templates = {
                VisualizationType.HISTOGRAM: self._generate_histogram_code,
                VisualizationType.BOXPLOT: self._generate_boxplot_code,
                VisualizationType.SCATTER_PLOT: self._generate_scatter_code,
                VisualizationType.LINE_PLOT: self._generate_line_plot_code,
                VisualizationType.BAR_CHART: self._generate_bar_chart_code,
                VisualizationType.CORRELATION_HEATMAP: self._generate_heatmap_code
            }

            if viz_type in viz_templates:
                return viz_templates[viz_type](columns, context)
            else:
                return self._generate_generic_visualization_code(viz_type, columns, context)

        except Exception as e:
            self.logger.error(f"Erro ao gerar código de visualização: {e}")
            return f"# Erro ao gerar visualização {viz_type.value}: {e}"

    def _generate_code_for_analysis(
        self,
        query_text: str,
        analysis_types: List[EDAAnalysisType],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Gera código para análises específicas usando LLM."""
        try:
            # Preparar contexto dos dados
            data_context = ""
            if context and "csv_metadata" in context:
                metadata = context["csv_metadata"]
                data_context = f"""
                Informações do dataset:
                - Colunas: {metadata.get('columns', [])}
                - Tipos de dados: {metadata.get('data_types', {})}
                - Shape: {metadata.get('shape', [0, 0])}
                """

            # Preparar prompt para geração de código
            system_prompt = """
            Você é um especialista em Python e análise exploratória de dados (EDA).
            Gere código Python que realize apenas análise exploratória, sem transformar os dados.

            Regras importantes:
            1. Use apenas bibliotecas padrão: pandas, numpy, matplotlib, seaborn, scipy
            2. NUNCA modifique ou transforme os dados originais
            3. Foque em análise descritiva e visualização
            4. Assuma que os dados estão na variável 'data' (pandas DataFrame)
            5. Use matplotlib.pyplot.show() para exibir gráficos
            6. Inclua comentários explicativos
            7. Trate possíveis erros (dados ausentes, tipos incorretos)

            Tipos de análise disponíveis:
            - DESCRIPTIVE: estatísticas descritivas, tipos de dados, completude
            - PATTERN: tendências, padrões temporais, agrupamentos
            - ANOMALY: detecção de outliers
            - RELATIONSHIP: correlações, relacionamentos entre variáveis

            Gere código limpo, eficiente e bem comentado.
            """

            user_prompt = f"""
            Consulta do usuário: "{query_text}"

            Tipos de análise necessários: {[t.value for t in analysis_types]}

            {data_context}

            Gere código Python que responda à consulta realizando as análises solicitadas.
            O código deve ser focado em EDA e não deve transformar os dados.
            """

            response = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
            )

            # Extrair código da resposta
            code = self._extract_code_from_response(response["content"])

            # Validar e limpar código
            validated_code = self._validate_and_clean_code(code)

            return validated_code

        except Exception as e:
            self.logger.warning(f"Erro ao gerar código com LLM: {e}")
            # Fallback para código básico
            return self._generate_fallback_code(query_text, analysis_types)

    def _extract_code_from_response(self, response_content: str) -> str:
        """Extrai código Python da resposta do LLM."""
        try:
            # Procurar por blocos de código Python
            import re

            # Padrão para blocos ```python ou ```
            python_pattern = r'```(?:python)?\s*(.*?)\s*```'
            matches = re.findall(python_pattern, response_content, re.DOTALL)

            if matches:
                return matches[0].strip()

            # Se não encontrar blocos, procurar linhas que parecem código
            lines = response_content.split('\n')
            code_lines = []

            for line in lines:
                line = line.strip()
                # Identificar linhas que parecem código Python
                if (line.startswith(('import ', 'from ', 'data.', 'df.', 'plt.', 'sns.', 'np.')) or
                    'print(' in line or '=' in line):
                    code_lines.append(line)

            if code_lines:
                return '\n'.join(code_lines)

            return response_content.strip()

        except Exception as e:
            self.logger.warning(f"Erro ao extrair código: {e}")
            return response_content

    def _validate_and_clean_code(self, code: str) -> str:
        """Valida e limpa o código gerado."""
        try:
            # Remover linhas perigosas ou que transformam dados
            dangerous_patterns = [
                'data.drop',
                'data.fillna',
                'data.replace',
                'data.loc[',
                'data.iloc[',
                'del data',
                'data =',
                'import os',
                'import sys',
                'os.',
                'sys.'
            ]

            lines = code.split('\n')
            clean_lines = []

            for line in lines:
                line_lower = line.lower().strip()

                # Pular linhas vazias e comentários
                if not line_lower or line_lower.startswith('#'):
                    clean_lines.append(line)
                    continue

                # Verificar padrões perigosos
                is_safe = True
                for pattern in dangerous_patterns:
                    if pattern.lower() in line_lower:
                        is_safe = False
                        self.logger.warning(f"Linha removida por segurança: {line.strip()}")
                        break

                if is_safe:
                    clean_lines.append(line)

            # Adicionar imports básicos se não estiverem presentes
            cleaned_code = '\n'.join(clean_lines)
            if 'import pandas' not in cleaned_code and 'import numpy' not in cleaned_code:
                imports = [
                    "import pandas as pd",
                    "import numpy as np",
                    "import matplotlib.pyplot as plt",
                    "import seaborn as sns",
                    "",
                    "# Código de análise EDA:",
                    ""
                ]
                cleaned_code = '\n'.join(imports) + cleaned_code

            return cleaned_code

        except Exception as e:
            self.logger.error(f"Erro ao validar código: {e}")
            return code

    def _generate_fallback_code(self, query_text: str, analysis_types: List[EDAAnalysisType]) -> str:
        """Gera código básico como fallback."""
        code_parts = [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "# Análise EDA básica",
            "print('Informações básicas do dataset:')",
            "print(f'Shape: {data.shape}')",
            "print(f'Colunas: {list(data.columns)}')",
            "print()",
            "print('Primeiras linhas:')",
            "print(data.head())",
            "print()",
            "print('Estatísticas descritivas:')",
            "print(data.describe())",
        ]

        if EDAAnalysisType.ANOMALY in analysis_types:
            code_parts.extend([
                "",
                "# Análise de outliers",
                "numeric_cols = data.select_dtypes(include=[np.number]).columns",
                "if len(numeric_cols) > 0:",
                "    for col in numeric_cols:",
                "        Q1 = data[col].quantile(0.25)",
                "        Q3 = data[col].quantile(0.75)",
                "        IQR = Q3 - Q1",
                "        outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]",
                "        print(f'Outliers em {col}: {len(outliers)}')"
            ])

        if EDAAnalysisType.RELATIONSHIP in analysis_types:
            code_parts.extend([
                "",
                "# Matriz de correlação",
                "numeric_data = data.select_dtypes(include=[np.number])",
                "if len(numeric_data.columns) > 1:",
                "    correlation_matrix = numeric_data.corr()",
                "    plt.figure(figsize=(10, 8))",
                "    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')",
                "    plt.title('Matriz de Correlação')",
                "    plt.show()"
            ])

        return '\n'.join(code_parts)

    def _generate_code_explanation(self, code: str, query_text: str) -> str:
        """Gera explicação do código gerado."""
        try:
            if not code:
                return "Nenhum código foi gerado."

            system_prompt = """
            Você é um especialista em Python que explica código de análise de dados.
            Forneça uma explicação clara e didática do código, focando em:
            1. O que cada seção faz
            2. Por que essas análises são relevantes
            3. Como interpretar os resultados

            Mantenha a explicação concisa mas informativa.
            """

            user_prompt = f"""
            Consulta original: "{query_text}"

            Código gerado:
            ```python
            {code}
            ```

            Explique este código de forma clara e didática.
            """

            response = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
            )

            return response["content"]

        except Exception as e:
            self.logger.warning(f"Erro ao gerar explicação: {e}")
            return "O código realiza análises exploratórias baseadas na consulta fornecida."

    def _execute_generated_code(self, code: str, context: Optional[Dict[str, Any]]):
        """Executa código gerado de forma segura."""
        try:
            # Preparar variáveis de contexto
            context_vars = {}
            if context and "dataframe" in context:
                context_vars["data"] = context["dataframe"]

            # Executar código
            execution_result = self.code_executor.execute_code(
                code=code,
                context_vars=context_vars,
                capture_output=True
            )

            return execution_result

        except Exception as e:
            self.logger.error(f"Erro ao executar código: {e}")
            return None

    # Métodos para gerar visualizações específicas
    def _generate_histogram_code(self, columns: List[str], context: Optional[Dict[str, Any]]) -> str:
        """Gera código para histograma."""
        if not columns:
            return "# Nenhuma coluna especificada para histograma"

        col = columns[0]
        return f"""
import matplotlib.pyplot as plt

# Histograma para {col}
plt.figure(figsize=(10, 6))
plt.hist(data['{col}'].dropna(), bins=30, alpha=0.7, edgecolor='black')
plt.title(f'Distribuição de {col}')
plt.xlabel('{col}')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)
plt.show()
"""

    def _generate_boxplot_code(self, columns: List[str], context: Optional[Dict[str, Any]]) -> str:
        """Gera código para boxplot."""
        if not columns:
            return "# Nenhuma coluna especificada para boxplot"

        col = columns[0]
        return f"""
import matplotlib.pyplot as plt

# Boxplot para {col}
plt.figure(figsize=(8, 6))
plt.boxplot(data['{col}'].dropna())
plt.title(f'Boxplot de {col}')
plt.ylabel('{col}')
plt.show()
"""

    def _generate_scatter_code(self, columns: List[str], context: Optional[Dict[str, Any]]) -> str:
        """Gera código para gráfico de dispersão."""
        if len(columns) < 2:
            return "# Necessárias pelo menos 2 colunas para gráfico de dispersão"

        col1, col2 = columns[0], columns[1]
        return f"""
import matplotlib.pyplot as plt

# Gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(data['{col1}'], data['{col2}'], alpha=0.6)
plt.title(f'Dispersão: {col1} vs {col2}')
plt.xlabel('{col1}')
plt.ylabel('{col2}')
plt.grid(True, alpha=0.3)
plt.show()
"""

    def _generate_line_plot_code(self, columns: List[str], context: Optional[Dict[str, Any]]) -> str:
        """Gera código para gráfico de linha."""
        if not columns:
            return "# Nenhuma coluna especificada para gráfico de linha"

        col = columns[0]
        return f"""
import matplotlib.pyplot as plt

# Gráfico de linha
plt.figure(figsize=(12, 6))
plt.plot(data['{col}'].dropna())
plt.title(f'Tendência de {col}')
plt.xlabel('Índice')
plt.ylabel('{col}')
plt.grid(True, alpha=0.3)
plt.show()
"""

    def _generate_bar_chart_code(self, columns: List[str], context: Optional[Dict[str, Any]]) -> str:
        """Gera código para gráfico de barras."""
        if not columns:
            return "# Nenhuma coluna especificada para gráfico de barras"

        col = columns[0]
        return f"""
import matplotlib.pyplot as plt

# Gráfico de barras (contagem de valores)
plt.figure(figsize=(10, 6))
value_counts = data['{col}'].value_counts().head(10)
value_counts.plot(kind='bar')
plt.title(f'Contagem de valores para {col}')
plt.xlabel('{col}')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.show()
"""

    def _generate_heatmap_code(self, columns: List[str], context: Optional[Dict[str, Any]]) -> str:
        """Gera código para heatmap de correlação."""
        return """
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap de correlação
numeric_data = data.select_dtypes(include=['number'])
if len(numeric_data.columns) > 1:
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.show()
else:
    print('Dados insuficientes para heatmap de correlação')
"""

    def _generate_generic_visualization_code(self, viz_type: VisualizationType, columns: List[str], context: Optional[Dict[str, Any]]) -> str:
        """Gera código genérico para visualizações."""
        return f"""
# Visualização genérica para {viz_type.value}
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Código de visualização personalizada baseado no tipo {viz_type.value}
plt.title(f'Visualização: {viz_type.value}')
plt.show()
"""


# Instância singleton
_code_generator_agent: Optional[CodeGeneratorAgent] = None


def get_code_generator_agent() -> CodeGeneratorAgent:
    """Obtém instância singleton do CodeGeneratorAgent."""
    global _code_generator_agent
    if _code_generator_agent is None:
        _code_generator_agent = CodeGeneratorAgent()
    return _code_generator_agent