"""
Gerenciador centralizado para interação com Large Language Models.
Fornece interface única para OpenAI, incluindo retry logic, rate limiting,
controle de custos e logging de uso para operações do sistema EDA.
"""

import openai
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import get_config
from .logger import get_logger, log_llm_call, log_error_with_context


@dataclass
class LLMUsageStats:
    """Estatísticas de uso do LLM."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_request: Optional[datetime] = None
    errors_count: int = 0


class LLMManager:
    """Gerenciador centralizado para operações com LLMs."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("llm_manager")
        self.usage_stats = LLMUsageStats()

        # Configurar cliente OpenAI
        openai.api_key = self.config.openai_api_key

        # Rate limiting (requests por minuto)
        self.rate_limit = 60
        self.request_times: List[datetime] = []

    def _check_rate_limit(self) -> None:
        """Verifica e aplica rate limiting."""
        now = datetime.now()
        # Remove requests mais antigos que 1 minuto
        self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]

        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0]).seconds
            self.logger.warning(f"Rate limit atingido. Aguardando {sleep_time}s")
            time.sleep(sleep_time)

    def _estimate_tokens(self, text: str) -> int:
        """Estimativa simples de tokens (aproximadamente 4 caracteres por token)."""
        return len(text) // 4

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calcula custo estimado baseado no modelo e tokens."""
        # Preços aproximados por 1K tokens (podem variar)
        pricing = {
            "gpt-5-nano-2025-08-07": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }

        model_pricing = pricing.get(model, pricing["gpt-5-nano-2025-08-07"])
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        return input_cost + output_cost

    def _retry_with_backoff(self, func, max_retries: int = 3, *args, **kwargs):
        """Executa função com retry e backoff exponencial."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                wait_time = (2 ** attempt) + 1
                self.logger.warning(f"Tentativa {attempt + 1} falhou. Aguardando {wait_time}s: {e}")
                time.sleep(wait_time)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa chat completion com retry e logging.

        Args:
            messages: Lista de mensagens no formato OpenAI
            model: Modelo a ser usado (padrão do config)
            max_tokens: Máximo de tokens na resposta
            system_prompt: Prompt de sistema opcional

        Returns:
            Dict contendo resposta e metadados
        """
        start_time = time.time()
        model = model or self.config.llm_model

        # Preparar mensagens
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Rate limiting
        self._check_rate_limit()
        self.request_times.append(datetime.now())

        try:
            # Executar com retry
            params = {
                "model": model,
                "messages": messages
            }
            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            response = self._retry_with_backoff(
                openai.chat.completions.create,
                **params
            )

            # Extrair dados da resposta
            content = response.choices[0].message.content
            usage = response.usage
            duration = time.time() - start_time

            # Calcular custos
            estimated_cost = self._calculate_cost(
                model,
                usage.prompt_tokens,
                usage.completion_tokens
            )

            # Atualizar estatísticas
            self._update_stats(usage.total_tokens, estimated_cost, duration)

            # Log da operação
            log_llm_call(
                model=model,
                tokens_used=usage.total_tokens,
                duration=duration
            )

            return {
                "content": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "model": model,
                "duration": duration,
                "estimated_cost": estimated_cost
            }

        except Exception as e:
            self.usage_stats.errors_count += 1
            log_error_with_context(
                error=e,
                context={
                    "model": model,
                    "messages_count": len(messages),
                }
            )
            raise

    def _update_stats(self, tokens: int, cost: float, duration: float) -> None:
        """Atualiza estatísticas de uso."""
        self.usage_stats.total_requests += 1
        self.usage_stats.total_tokens += tokens
        self.usage_stats.total_cost += cost
        self.usage_stats.last_request = datetime.now()

        # Atualizar média de tempo de resposta
        total_time = self.usage_stats.avg_response_time * (self.usage_stats.total_requests - 1)
        self.usage_stats.avg_response_time = (total_time + duration) / self.usage_stats.total_requests

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Gera texto simples a partir de prompt.

        Args:
            prompt: Texto do prompt
            model: Modelo a ser usado
            max_tokens: Máximo de tokens na resposta

        Returns:
            Texto gerado
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens
        )
        return response["content"]

    def analyze_query(
        self,
        user_query: str,
        csv_info: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analisa intenção da consulta do usuário.

        Args:
            user_query: Pergunta do usuário
            csv_info: Informações sobre os dados CSV
            context: Contexto adicional opcional

        Returns:
            Análise estruturada da consulta
        """
        system_prompt = """
        Você é um especialista em análise exploratória de dados (EDA).
        Analise a consulta do usuário e determine:
        1. Tipo de análise necessária
        2. Colunas envolvidas
        3. Tipo de visualização recomendada
        4. Complexidade estimada

        Responda em formato JSON estruturado.
        """

        user_prompt = f"""
        Consulta do usuário: "{user_query}"

        Informações dos dados:
        {csv_info}

        {f'Contexto adicional: {context}' if context else ''}

        Analise a consulta e forneça uma resposta estruturada.
        """

        response = self.chat_completion(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        )

        return response

    def generate_analysis_code(
        self,
        analysis_type: str,
        columns: List[str],
        data_info: str,
        requirements: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gera código Python para análise específica.

        Args:
            analysis_type: Tipo de análise (descriptive, correlation, etc.)
            columns: Colunas envolvidas na análise
            data_info: Informações sobre os dados
            requirements: Requisitos específicos

        Returns:
            Código gerado e explicações
        """
        system_prompt = """
        Você é um especialista em Python e análise de dados.
        Gere código Python limpo e eficiente para análise exploratória.
        Use pandas, numpy, matplotlib e seaborn quando apropriado.
        Inclua tratamento de erros e validações básicas.
        """

        user_prompt = f"""
        Gere código Python para:
        - Tipo de análise: {analysis_type}
        - Colunas: {columns}
        - Dados: {data_info}
        {f'- Requisitos: {requirements}' if requirements else ''}

        Retorne código executável e comentários explicativos.
        """

        response = self.chat_completion(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        )

        return response

    def get_tool_calls(
        self,
        user_query: str,
        dataset_info: Dict[str, Any],
        available_tools: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém do LLM a lista de ferramentas a serem chamadas para responder a query.

        Args:
            user_query: Query do usuário
            dataset_info: Informações sobre o dataset
            available_tools: Lista de ferramentas disponíveis
            context: Contexto adicional

        Returns:
            Lista de chamadas de ferramentas no formato:
            [{"tool_name": str, "parameters": dict}, ...]
        """
        import json
        import re

        system_prompt = """
Você é um orquestrador de ferramentas estatísticas para análise exploratória de dados.

Sua tarefa é analisar a query do usuário e decidir quais ferramentas estatísticas devem ser executadas
para responder à pergunta de forma precisa e completa.

CLASSIFICAÇÃO DE QUERIES:

1. QUERIES VISUAIS (use ferramentas de visualização - PRIORIDADE MÁXIMA):
   - "Qual a distribuição de X?" → plot_distribution
   - "Mostre a distribuição de Y" → plot_distribution
   - "Histograma de Z" → plot_distribution
   - "Gráfico de W" → plot_distribution
   - "Como está distribuído A?" → plot_distribution
   - "Boxplot de B" → plot_boxplot
   - "Outliers de C" → plot_boxplot
   - "Diagrama de caixa de D" → plot_boxplot
   - "Correlação entre variáveis" → plot_correlation_heatmap
   - "Matriz de correlação" → plot_correlation_heatmap
   - "Heatmap" → plot_correlation_heatmap
   - "Gráfico de barras de E" → plot_bar_chart (para variáveis categóricas)

   IMPORTANTE: Se a query mencionar "distribuição", "gráfico", "plot", "mostre", "visualize",
   SEMPRE use ferramentas de visualização (plot_*), NÃO use ferramentas estatísticas.

2. QUERIES CONCEITUAIS/DESCRITIVAS (use analyze_dataset_schema):
   - "Quais são os tipos de dados?" → analyze_dataset_schema
   - "Quais colunas são numéricas?" → analyze_dataset_schema
   - "Quais variáveis são categóricas?" → analyze_dataset_schema
   - "Quantas colunas tem o dataset?" → analyze_dataset_schema
   - "Qual a estrutura do dataset?" → analyze_dataset_schema
   - "Liste as variáveis booleanas" → analyze_dataset_schema

3. QUERIES NUMÉRICAS ESPECÍFICAS (use ferramentas minimalistas):
   - "Qual a média de X?" → get_mean
   - "Qual o máximo de Y?" → get_max
   - "Qual o mínimo de Z?" → get_min
   - "Qual a mediana de W?" → get_median
   - "Qual a soma de A?" → get_sum
   - "Qual o desvio padrão de B?" → get_std
   - "Quantos valores tem C?" → get_count

4. QUERIES ANALÍTICAS COMPLEXAS:
   - "Há correlação entre X e Y?" → calculate_correlation_matrix (retorna apenas números)
   - "Quais são os outliers?" → detect_outliers_iqr
   - "Há dados ausentes?" → analyze_missing_data

PRINCÍPIO DA MINIMALIDADE (CRÍTICO):
- Use SEMPRE a ferramenta MAIS ESPECÍFICA disponível para a query
- Se o usuário pede APENAS "média", use get_mean (NÃO use calculate_descriptive_stats)
- Se a query é conceitual (tipos, estrutura), use analyze_dataset_schema
- Se a query é numérica específica, use ferramenta minimalista (get_mean, get_max, etc.)

Use calculate_descriptive_stats APENAS se:
- O usuário pedir "análise completa", "todas as estatísticas", "estatísticas descritivas completas"
- O usuário pedir múltiplos valores E não existir ferramenta específica para cada um

OUTRAS REGRAS:
- Use apenas as ferramentas fornecidas na lista
- Não invente resultados, apenas indique quais ferramentas executar
- Você pode chamar múltiplas ferramentas se o usuário pedir múltiplos valores
- Seja preciso nos parâmetros (especialmente nomes de colunas)
- Se a query é conceitual e NÃO requer cálculos, você pode retornar tool_calls vazio []
  (o sistema ativará modo conceitual automaticamente)

Responda SEMPRE em formato JSON com esta estrutura:
{
  "reasoning": "Breve explicação do raciocínio",
  "tool_calls": [
    {
      "tool_name": "nome_da_ferramenta",
      "parameters": {
        "param1": "valor1",
        "param2": "valor2"
      }
    }
  ]
}
"""

        # Formatar informações do dataset
        dataset_summary = f"""
Dataset Info:
- Shape: {dataset_info.get('shape', 'N/A')}
- Colunas: {dataset_info.get('columns', [])}
- Tipos de dados: {dataset_info.get('dtypes', {})}
"""

        # Formatar ferramentas disponíveis
        tools_description = "Ferramentas Disponíveis:\n\n"
        for tool in available_tools:
            tools_description += f"Nome: {tool['name']}\n"
            tools_description += f"Descrição: {tool['description']}\n"
            tools_description += "Parâmetros:\n"
            for param in tool['parameters']:
                required_str = " (obrigatório)" if param['required'] else " (opcional)"
                tools_description += f"  - {param['name']} ({param['type']}){required_str}: {param['description']}\n"
            tools_description += "\n"

        user_prompt = f"""
Query do Usuário: "{user_query}"

{dataset_summary}

{tools_description}

{f'Contexto Adicional: {context}' if context else ''}

Analise a query e retorne em JSON as ferramentas que devem ser executadas.
"""

        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )

            content = response["content"]

            # Tentar extrair JSON do conteúdo
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Tentar encontrar JSON sem markdown
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content

            parsed = json.loads(json_str)

            # Extrair chamadas de ferramentas
            tool_calls = parsed.get('tool_calls', [])

            self.logger.info(f"LLM sugeriu {len(tool_calls)} ferramentas: {[t['tool_name'] for t in tool_calls]}")

            return tool_calls

        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao parsear JSON do LLM: {e}")
            self.logger.debug(f"Conteúdo recebido: {content}")
            # Fallback: tentar detectar query simples
            return self._fallback_tool_detection(user_query, dataset_info)
        except Exception as e:
            self.logger.error(f"Erro ao obter tool calls: {e}")
            return self._fallback_tool_detection(user_query, dataset_info)

    def _fallback_tool_detection(self, user_query: str, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback para detecção de ferramentas usando heurísticas simples.

        Args:
            user_query: Query do usuário
            dataset_info: Informações do dataset

        Returns:
            Lista de tool calls
        """
        query_lower = user_query.lower()
        columns = dataset_info.get('columns', [])

        # Detectar coluna mencionada
        target_column = None
        for col in columns:
            if col.lower() in query_lower:
                target_column = col
                break

        # Detectar tipo de análise - usar ferramentas específicas
        if any(keyword in query_lower for keyword in ['média', 'mean', 'average']):
            if target_column:
                return [{
                    "tool_name": "get_mean",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['máximo', 'max', 'maximum', 'maior']):
            if target_column:
                return [{
                    "tool_name": "get_max",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['mínimo', 'min', 'minimum', 'menor']):
            if target_column:
                return [{
                    "tool_name": "get_min",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['mediana', 'median']):
            if target_column:
                return [{
                    "tool_name": "get_median",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['soma', 'sum', 'total']) and 'linha' not in query_lower:
            if target_column:
                return [{
                    "tool_name": "get_sum",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['desvio padrão', 'std', 'standard deviation']):
            if target_column:
                return [{
                    "tool_name": "get_std",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['contagem', 'count', 'quantos']):
            if target_column:
                return [{
                    "tool_name": "get_count",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['correlação', 'correlation', 'relacionamento']):
            return [{
                "tool_name": "calculate_correlation_matrix",
                "parameters": {}
            }]

        if any(keyword in query_lower for keyword in ['outlier', 'anomalia', 'atípico']):
            if target_column:
                return [{
                    "tool_name": "detect_outliers_iqr",
                    "parameters": {"column": target_column}
                }]

        if any(keyword in query_lower for keyword in ['missing', 'ausente', 'faltante', 'nulo']):
            return [{
                "tool_name": "analyze_missing_data",
                "parameters": {}
            }]

        # Default: análise de dados ausentes (mais útil que estatísticas completas)
        return [{
            "tool_name": "analyze_missing_data",
            "parameters": {}
        }]

    def synthesize_tool_results(
        self,
        user_query: str,
        tool_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Sintetiza resultados das ferramentas em resposta em linguagem natural.

        Args:
            user_query: Query original do usuário
            tool_results: Resultados das ferramentas executadas
            context: Contexto de memória da sessão (dict com histórico de queries e insights)

        Returns:
            Resposta sintetizada em linguagem natural
        """
        system_prompt = """
Você é um especialista em análise exploratória de dados que explica resultados de forma clara e didática.

Sua tarefa é interpretar os resultados das ferramentas estatísticas e apresentar uma resposta
natural, interpretativa e instrutiva, indo além da simples devolução de números ou tabelas.

## ESTRUTURA DA RESPOSTA

Toda resposta deve seguir esta estrutura em 4 partes:

1. **RESPOSTA DIRETA** - Responda clara e especificamente à pergunta do usuário
2. **INTERPRETAÇÃO** - Explique o que os resultados significam na prática
3. **MÉTODO UTILIZADO** - Descreva brevemente o método/critério utilizado para a análise
4. **RECOMENDAÇÕES** - Sugira possíveis próximos passos ou ações com base no resultado

## REGRAS FUNDAMENTAIS

- Use os valores EXATOS retornados pelas ferramentas - NUNCA invente ou aproxime números
- Use formatação markdown para destacar informações importantes (negrito para valores-chave)
- Mantenha tom profissional, acessível e didático
- Adapte a explicação ao contexto da pergunta e tipo de análise
- Organize insights em tópicos quando houver múltiplas informações

## EXEMPLOS POR TIPO DE ANÁLISE

### Exemplo 1 - Detecção de Outliers:
Query: "Existem valores atípicos nos dados?"

Resposta:
**Resultado:** Foram detectados valores atípicos em várias variáveis do dataset.

**Principais Colunas Afetadas:**
- **V27**: 39.015 outliers (13,7% dos dados)
- **Amount**: 31.881 outliers (11,2% dos dados)
- **V14**, **V12**, **V10**: também apresentam concentração significativa

**Método Utilizado:**
Aplicamos o método IQR (Intervalo Interquartil), que identifica valores que estão muito distantes
da maior parte dos dados - especificamente, valores além de 1,5× IQR dos quartis Q1 e Q3.

**Impacto e Recomendações:**
- Outliers em alta concentração (>10%) podem distorcer médias e modelos preditivos
- **Investigue** se são erros de medição ou valores extremos legítimos
- **Opções de tratamento:** transformação logarítmica, remoção seletiva, ou modelagem separada
- Para variáveis financeiras como Amount, valores extremos podem indicar fraudes ou transações VIP

### Exemplo 2 - Estatísticas Descritivas:
Query: "Qual a média da coluna Amount?"

Resposta:
**Resultado:** A média da coluna Amount é **88,35**.

**Interpretação:**
Este valor indica que o valor médio das transações está em torno de 88,35 unidades. No entanto,
para dados financeiros, é importante considerar que a média pode ser influenciada por valores extremos.

**Contexto Estatístico:**
A média é calculada somando todos os valores e dividindo pelo número de observações. É sensível
a outliers - valores muito altos ou baixos podem puxar a média para cima ou para baixo.

**Recomendações:**
- Compare com a **mediana** para verificar se há assimetria na distribuição
- Verifique o **desvio padrão** para entender a variabilidade dos dados
- Analise a presença de **outliers** que possam estar distorcendo a média

### Exemplo 3 - Correlações:
Query: "Existe correlação entre X e Y?"

Resposta:
**Resultado:** Foi identificada uma correlação **forte e positiva** de 0,85 entre as variáveis X e Y.

**Interpretação:**
Uma correlação de 0,85 indica que quando X aumenta, Y tende a aumentar também de forma consistente.
Essa é uma relação linear forte (valores acima de 0,7 são considerados fortes).

**Método Utilizado:**
Calculamos o coeficiente de correlação de Pearson, que mede a força e direção da relação linear
entre duas variáveis numéricas. Valores variam de -1 (correlação negativa perfeita) a +1 (positiva perfeita).

**Implicações e Próximos Passos:**
- Esta forte correlação sugere que X pode ser um bom preditor para Y
- **Atenção:** correlação não implica causalidade - pode haver fatores externos influenciando ambas
- **Considere:** análise de regressão para modelar a relação quantitativamente
- **Verifique:** se há multicolinearidade caso esteja construindo modelos preditivos

## FORMATAÇÃO E TOM

- Use seções com títulos em negrito para organizar a informação
- Destaque valores e métricas importantes com **negrito**
- Use bullet points para listar múltiplos insights
- Mantenha parágrafos curtos e objetivos
- Evite jargão excessivo, mas seja tecnicamente preciso
"""

        import json

        # Detectar tipo de análise baseado nas ferramentas executadas
        analysis_type = "geral"
        tool_categories = set()

        for tool_result in tool_results:
            tool_name = tool_result.get('tool_name', '').lower()

            if 'outlier' in tool_name or 'anomaly' in tool_name:
                tool_categories.add('outlier_detection')
            elif 'correlation' in tool_name or 'relationship' in tool_name:
                tool_categories.add('correlation')
            elif 'stats' in tool_name or 'statistics' in tool_name or 'descriptive' in tool_name:
                tool_categories.add('statistics')
            elif 'missing' in tool_name or 'null' in tool_name:
                tool_categories.add('missing_data')
            elif 'distribution' in tool_name or 'histogram' in tool_name:
                tool_categories.add('distribution')

        # Determinar tipo predominante
        if 'outlier_detection' in tool_categories:
            analysis_type = "outlier_detection"
        elif 'correlation' in tool_categories:
            analysis_type = "correlation"
        elif 'statistics' in tool_categories:
            analysis_type = "statistics"
        elif 'missing_data' in tool_categories:
            analysis_type = "missing_data"
        elif 'distribution' in tool_categories:
            analysis_type = "distribution"

        # Formatar resultados das ferramentas
        tools_summary = "Resultados das Ferramentas Executadas:\n\n"
        for idx, tool_result in enumerate(tool_results, 1):
            tool_name = tool_result.get('tool_name', 'Unknown')
            parameters = tool_result.get('parameters', {})
            result = tool_result.get('result', {})

            tools_summary += f"Ferramenta {idx}: {tool_name}\n"
            tools_summary += f"Parâmetros: {parameters}\n"

            if result.get('success', False):
                tools_summary += f"Status: Sucesso\n"
                tools_summary += f"Dados:\n{json.dumps(result.get('data', {}), indent=2, ensure_ascii=False)}\n"
            else:
                tools_summary += f"Status: Erro\n"
                tools_summary += f"Erro: {result.get('error', 'Unknown error')}\n"

            tools_summary += "\n"

        # Adicionar dica contextual baseada no tipo de análise
        context_hints = {
            'outlier_detection': "Lembre-se de explicar o impacto dos outliers e sugerir tratamentos apropriados.",
            'correlation': "Lembre-se de explicar a força da correlação e alertar sobre causalidade vs correlação.",
            'statistics': "Lembre-se de contextualizar as estatísticas e sugerir análises complementares.",
            'missing_data': "Lembre-se de explicar o impacto dos dados ausentes e sugerir estratégias de tratamento.",
            'distribution': "Lembre-se de caracterizar a forma da distribuição e suas implicações."
        }

        analysis_hint = context_hints.get(analysis_type, "Forneça uma resposta completa e contextualizada.")

        # Formatar contexto de memória se disponível
        memory_context_section = ""
        if context and isinstance(context, dict):
            memory_context_section = "\n## CONTEXTO DA SESSÃO (Memória de Interações Anteriores)\n\n"

            # Adicionar queries recentes
            if context.get('recent_queries'):
                memory_context_section += "**Perguntas Recentes do Usuário:**\n"
                for i, query in enumerate(context['recent_queries'][-5:], 1):
                    memory_context_section += f"{i}. {query}\n"
                memory_context_section += "\n"

            # Adicionar insights de alta confiança
            if context.get('high_confidence_insights'):
                memory_context_section += "**Insights Anteriores (Alta Confiança):**\n"
                for i, insight in enumerate(context['high_confidence_insights'][:5], 1):
                    memory_context_section += f"{i}. {insight}\n"
                memory_context_section += "\n"

            # Adicionar estatísticas da sessão
            if context.get('total_queries'):
                memory_context_section += f"**Estatísticas da Sessão:** {context['total_queries']} queries anteriores, "
                memory_context_section += f"{context.get('total_insights', 0)} insights gerados\n\n"

        user_prompt = f"""
Query Original do Usuário: "{user_query}"

Tipo de Análise Detectado: {analysis_type}

{tools_summary}
{memory_context_section}

Com base nos resultados das ferramentas acima, forneça uma resposta completa e estruturada seguindo o formato estabelecido:

1. **Responda diretamente** à pergunta do usuário com os valores exatos
2. **Interprete** o que esses resultados significam na prática
3. **Explique brevemente** o método ou critério utilizado pela ferramenta
4. **Forneça recomendações** sobre possíveis próximos passos ou ações

Use formatação markdown para organizar a resposta e destacar informações importantes.

DICA CONTEXTUAL: {analysis_hint}
"""

        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )

            return response["content"]

        except Exception as e:
            self.logger.error(f"Erro ao sintetizar resultados: {e}")
            # Fallback: resposta básica
            return f"Ferramentas executadas com sucesso. Verifique os resultados detalhados acima."

    def conceptual_synthesis(
        self,
        user_query: str,
        dataset_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Sintetiza resposta conceitual usando apenas informações do schema do dataset.

        Este método é usado quando não há ferramentas estatísticas aplicáveis,
        mas a query pode ser respondida com base na estrutura do dataset.

        Args:
            user_query: Query original do usuário
            dataset_info: Informações estruturais do dataset (shape, colunas, tipos)
            context: Contexto de memória da sessão (dict com histórico de queries e insights)

        Returns:
            Resposta sintetizada em linguagem natural
        """
        import json

        system_prompt = """
Você é um especialista em análise exploratória de dados.

Sua tarefa é responder queries CONCEITUAIS e DESCRITIVAS sobre datasets usando
apenas as informações estruturais fornecidas (schema, tipos de dados, colunas).

MODO CONCEITUAL - REGRAS:
1. Use APENAS as informações do schema fornecidas - NÃO invente dados
2. Responda de forma clara, direta e em linguagem natural
3. Classifique variáveis em categorias conceituais quando relevante:
   - Numéricas contínuas (float64, medidas contínuas)
   - Numéricas discretas (int64, contagens)
   - Categóricas (object, string, valores discretos)
   - Booleanas (bool, binário 0/1)
   - Temporais (datetime, datas)
4. NÃO calcule estatísticas - apenas descreva a estrutura
5. Se a query pedir cálculos numéricos, informe que ferramentas estatísticas são necessárias

EXEMPLOS DE QUERIES CONCEITUAIS:
- "Quais são os tipos de dados?"
- "Quais colunas são numéricas?"
- "Quais variáveis são categóricas?"
- "Quantas colunas tem o dataset?"
- "Liste as variáveis booleanas"
- "Qual a estrutura do dataset?"

FORMATO DE RESPOSTA:
- Seja conciso e direto
- Use listas quando listar múltiplas variáveis
- Organize por categorias quando aplicável
"""

        # Formatar informações do dataset
        dataset_summary = f"""
Informações do Dataset:
- Shape: {dataset_info.get('shape', 'N/A')}
- Total de linhas: {dataset_info.get('total_rows', 'N/A')}
- Total de colunas: {dataset_info.get('total_columns', 'N/A')}
- Colunas: {dataset_info.get('columns', [])}
- Tipos de dados: {json.dumps(dataset_info.get('dtypes', {}), indent=2, ensure_ascii=False)}
"""

        # Formatar contexto de memória se disponível
        memory_context_section = ""
        if context and isinstance(context, dict):
            memory_context_section = "\n## CONTEXTO DA SESSÃO (Memória de Interações Anteriores)\n\n"

            # Adicionar queries recentes
            if context.get('recent_queries'):
                memory_context_section += "**Perguntas Recentes do Usuário:**\n"
                for i, query in enumerate(context['recent_queries'][-5:], 1):
                    memory_context_section += f"{i}. {query}\n"
                memory_context_section += "\n"

            # Adicionar insights de alta confiança
            if context.get('high_confidence_insights'):
                memory_context_section += "**Insights Anteriores (Alta Confiança):**\n"
                for i, insight in enumerate(context['high_confidence_insights'][:5], 1):
                    memory_context_section += f"{i}. {insight}\n"
                memory_context_section += "\n"

            memory_context_section += "**IMPORTANTE:** Se a pergunta faz referência a análises anteriores (ex: 'quais conclusões você obteve?', 'o que você descobriu?'), USE as informações do histórico acima para responder. Se a pergunta for sobre dados específicos que você ainda não analisou, informe isso claramente.\n\n"

        user_prompt = f"""
Query do Usuário: "{user_query}"

{dataset_summary}
{memory_context_section}

Com base nas informações estruturais e no contexto da sessão acima, forneça uma resposta clara para a pergunta do usuário.
"""

        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )

            return response["content"]

        except Exception as e:
            self.logger.error(f"Erro ao sintetizar resposta conceitual: {e}")
            # Fallback básico
            return f"Não foi possível gerar resposta conceitual. Dataset possui {dataset_info.get('total_columns', 0)} colunas."

    def get_usage_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso do LLM."""
        return {
            "total_requests": self.usage_stats.total_requests,
            "total_tokens": self.usage_stats.total_tokens,
            "total_cost": round(self.usage_stats.total_cost, 4),
            "avg_response_time": round(self.usage_stats.avg_response_time, 2),
            "last_request": self.usage_stats.last_request.isoformat() if self.usage_stats.last_request else None,
            "errors_count": self.usage_stats.errors_count,
            "current_model": self.config.llm_model
        }

    def reset_stats(self) -> None:
        """Reseta estatísticas de uso."""
        self.usage_stats = LLMUsageStats()
        self.logger.info("Estatísticas de uso resetadas")


# Instância singleton para uso global
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Obtém instância singleton do LLMManager."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager