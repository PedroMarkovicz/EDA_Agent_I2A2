"""
Interpretador inteligente de consultas de usuário para análise EDA.
Utiliza LLM configurado para compreender intenções, extrair entidades,
classificar tipos de análise necessários e gerar plano de execução
estruturado para o sistema multi-agente.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .config import get_config
from .llm_manager import get_llm_manager
from .logger import get_logger, log_analysis_step, log_error_with_context
from ..models.query_schema import UserQuery, QueryClassification, QueryEntity
from ..models.enums import QueryIntentType, EDAAnalysisType, VisualizationType


class QueryParsingError(Exception):
    """Erro específico para parsing de consultas."""
    pass


class QueryInterpreter:
    """Interpretador inteligente para consultas de usuário."""

    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        self.logger = get_logger("query_interpreter")

        # Padrões regex para extração de entidades
        self.entity_patterns = {
            "column_name": r'\b(?:coluna|column|campo|field)\s+["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?',
            "numeric_value": r'\b\d+\.?\d*\b',
            "comparison": r'\b(maior|menor|igual|acima|abaixo|entre)\b',
            "aggregation": r'\b(média|mean|soma|sum|count|contagem|máximo|max|mínimo|min)\b'
        }

        # Mapeamentos de intenção
        self.intent_keywords = {
            QueryIntentType.DATA_OVERVIEW: [
                "visão geral", "overview", "resumo", "summary", "descrever", "describe",
                "mostrar dados", "show data", "explorar", "explore"
            ],
            QueryIntentType.STATISTICAL_SUMMARY: [
                "estatísticas", "statistics", "estatísticas descritivas", "descriptive statistics",
                "média", "mean", "mediana", "median", "desvio padrão", "standard deviation"
            ],
            QueryIntentType.CORRELATION_ANALYSIS: [
                "correlação", "correlation", "relacionamento", "relationship",
                "associação", "association", "dependência", "dependency"
            ],
            QueryIntentType.DISTRIBUTION_ANALYSIS: [
                "distribuição", "distribution", "histograma", "histogram",
                "densidade", "density", "frequência", "frequency"
            ],
            QueryIntentType.OUTLIER_DETECTION: [
                "outliers", "anomalias", "anomalies", "valores extremos", "extreme values",
                "discrepantes", "atípicos"
            ],
            QueryIntentType.TREND_ANALYSIS: [
                "tendência", "trend", "padrão", "pattern", "temporal", "time series",
                "evolução", "evolution", "mudança", "change"
            ],
            QueryIntentType.COMPARISON_ANALYSIS: [
                "comparar", "compare", "diferença", "difference", "contraste", "contrast",
                "versus", "vs", "entre", "between"
            ],
            QueryIntentType.SPECIFIC_VALUE_QUERY: [
                "qual", "what", "valor", "value", "máximo", "max", "maximum", "mínimo", "min", "minimum",
                "maior", "largest", "menor", "smallest", "total", "sum", "soma", "contagem", "count",
                "quantos", "how many", "quanto", "how much"
            ]
        }

        # Mapeamentos de visualização
        self.visualization_keywords = {
            VisualizationType.HISTOGRAM: ["histograma", "histogram", "distribuição", "distribution"],
            VisualizationType.BOXPLOT: ["boxplot", "box plot", "caixa", "quartis", "quartiles"],
            VisualizationType.SCATTER_PLOT: ["scatter", "dispersão", "scatter plot", "correlação"],
            VisualizationType.LINE_PLOT: ["linha", "line", "temporal", "time", "evolução"],
            VisualizationType.BAR_CHART: ["barras", "bar", "contagem", "count", "categorias"],
            VisualizationType.CORRELATION_HEATMAP: ["heatmap", "mapa de calor", "correlação", "correlation matrix"]
        }

    def interpret_query(
        self,
        query_text: str,
        csv_metadata: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> QueryClassification:
        """
        Interpreta consulta do usuário e retorna classificação estruturada.

        Args:
            query_text: Texto da consulta do usuário
            csv_metadata: Metadados do CSV carregado
            context: Contexto adicional da conversa

        Returns:
            QueryClassification com análise completa da consulta

        Raises:
            QueryParsingError: Se não conseguir interpretar a consulta
        """
        try:
            log_analysis_step("query_interpretation", "started", {"query_length": len(query_text)})

            # Extrair entidades básicas
            entities = self._extract_entities(query_text, csv_metadata)

            # Classificar intenção primária
            intent_type = self._classify_intent(query_text)

            # Determinar tipos de análise necessários
            analysis_types = self._determine_analysis_types(query_text, intent_type)

            # Detectar se precisa de visualização
            requires_visualization, viz_type = self._detect_visualization_needs(query_text)

            # Criar UserQuery
            user_query = UserQuery(
                query_text=query_text,
                intent_type=intent_type,
                analysis_types=analysis_types,
                entities=entities,
                requires_visualization=requires_visualization,
                visualization_type=viz_type,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "context": context
                }
            )

            # Usar LLM para análise mais sofisticada
            llm_analysis = self._get_llm_analysis(query_text, csv_metadata, context)

            # Consolidar resultados
            classification = self._consolidate_classification(user_query, llm_analysis)

            log_analysis_step(
                "query_interpretation", "completed",
                {
                    "intent": classification.query.intent_type.value if classification.query.intent_type else "unknown",
                    "analysis_types_count": len(classification.required_agents),
                    "confidence": classification.confidence_score
                }
            )

            return classification

        except Exception as e:
            log_error_with_context(e, {"query": query_text, "operation": "query_interpretation"})
            raise QueryParsingError(f"Erro ao interpretar consulta: {e}")

    def _extract_entities(self, query_text: str, csv_metadata: Optional[Dict[str, Any]]) -> List[QueryEntity]:
        """Extrai entidades da consulta usando regex e metadados."""
        entities = []
        query_lower = query_text.lower()

        # Extrair colunas mencionadas
        if csv_metadata and "columns" in csv_metadata:
            for column in csv_metadata["columns"]:
                if column.lower() in query_lower:
                    entities.append(QueryEntity(
                        name=column,
                        entity_type="column",
                        value=column
                    ))

        # Extrair valores numéricos
        numeric_matches = re.findall(self.entity_patterns["numeric_value"], query_text)
        for value in numeric_matches:
            entities.append(QueryEntity(
                name="numeric_value",
                entity_type="value",
                value=value
            ))

        # Extrair operadores de comparação
        comparison_matches = re.findall(self.entity_patterns["comparison"], query_lower)
        for comp in comparison_matches:
            entities.append(QueryEntity(
                name="comparison_operator",
                entity_type="operator",
                value=comp
            ))

        return entities

    def _classify_intent(self, query_text: str) -> Optional[QueryIntentType]:
        """Classifica a intenção principal da consulta."""
        query_lower = query_text.lower()

        intent_scores = {}

        for intent_type, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Peso maior para matches exatos
                    if f" {keyword} " in f" {query_lower} ":
                        score += 2
                    else:
                        score += 1

            if score > 0:
                intent_scores[intent_type] = score

        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]

        return None

    def _determine_analysis_types(self, query_text: str, intent_type: Optional[QueryIntentType]) -> List[EDAAnalysisType]:
        """Determina tipos de análise necessários baseado na consulta."""
        analysis_types = []
        query_lower = query_text.lower()

        # Mapear intenções para tipos de análise
        intent_to_analysis = {
            QueryIntentType.DATA_OVERVIEW: [EDAAnalysisType.DESCRIPTIVE],
            QueryIntentType.STATISTICAL_SUMMARY: [EDAAnalysisType.DESCRIPTIVE],
            QueryIntentType.CORRELATION_ANALYSIS: [EDAAnalysisType.RELATIONSHIP],
            QueryIntentType.DISTRIBUTION_ANALYSIS: [EDAAnalysisType.DESCRIPTIVE, EDAAnalysisType.PATTERN],
            QueryIntentType.OUTLIER_DETECTION: [EDAAnalysisType.ANOMALY],
            QueryIntentType.TREND_ANALYSIS: [EDAAnalysisType.PATTERN],
            QueryIntentType.COMPARISON_ANALYSIS: [EDAAnalysisType.DESCRIPTIVE, EDAAnalysisType.RELATIONSHIP],
            QueryIntentType.SPECIFIC_VALUE_QUERY: [EDAAnalysisType.DESCRIPTIVE]
        }

        if intent_type and intent_type in intent_to_analysis:
            analysis_types.extend(intent_to_analysis[intent_type])

        # Palavras-chave específicas para tipos de análise
        analysis_keywords = {
            EDAAnalysisType.DESCRIPTIVE: ["estatísticas", "resumo", "média", "mediana", "desvio", "máximo", "mínimo", "total", "soma", "contagem"],
            EDAAnalysisType.PATTERN: ["padrão", "tendência", "temporal", "evolução"],
            EDAAnalysisType.ANOMALY: ["outliers", "anomalias", "atípicos", "discrepantes"],
            EDAAnalysisType.RELATIONSHIP: ["correlação", "relacionamento", "associação", "dependência"]
        }

        for analysis_type, keywords in analysis_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if analysis_type not in analysis_types:
                    analysis_types.append(analysis_type)

        # Se nenhum tipo específico foi detectado, usar descriptive como padrão
        if not analysis_types:
            analysis_types = [EDAAnalysisType.DESCRIPTIVE]

        return analysis_types

    def _detect_visualization_needs(self, query_text: str) -> Tuple[bool, Optional[VisualizationType]]:
        """Detecta se a consulta requer visualização e qual tipo."""
        query_lower = query_text.lower()

        # Palavras que indicam necessidade de visualização
        viz_indicators = ["gráfico", "plot", "chart", "visualizar", "mostrar", "exibir", "desenhar"]

        requires_viz = any(indicator in query_lower for indicator in viz_indicators)

        # Detectar tipo específico de visualização
        viz_type = None
        viz_scores = {}

        for viz_type_enum, keywords in self.visualization_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                viz_scores[viz_type_enum] = score

        if viz_scores:
            viz_type = max(viz_scores.items(), key=lambda x: x[1])[0]
            requires_viz = True

        return requires_viz, viz_type

    def _get_llm_analysis(
        self,
        query_text: str,
        csv_metadata: Optional[Dict[str, Any]],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Obtem análise mais sofisticada usando LLM."""
        try:
            system_prompt = """
            Você é um especialista em análise exploratória de dados (EDA).
            Analise a consulta do usuário e forneça uma resposta estruturada em JSON com:

            {
                "confidence_score": float (0-1),
                "intent_type": string,
                "analysis_types": [string],
                "complexity": string ("low", "medium", "high"),
                "requires_visualization": boolean,
                "visualization_type": string,
                "entities": [{"name": string, "type": string, "value": string}],
                "processing_order": [string],
                "recommendations": [string]
            }

            IMPORTANTE:
            - Se a consulta é sobre um valor específico (máximo, mínimo, média, etc.), marque como "specific_value_query"
            - Identifique a coluna específica mencionada na consulta
            - Seja preciso e considere o contexto dos dados disponíveis
            """

            csv_info = ""
            if csv_metadata:
                csv_info = f"""
                Informações dos dados CSV:
                - Colunas: {csv_metadata.get('columns', [])}
                - Linhas: {csv_metadata.get('shape', [0, 0])[0]}
                - Tipos de dados: {csv_metadata.get('dtypes', {})}
                """

            user_prompt = f"""
            Consulta do usuário: "{query_text}"

            {csv_info}

            {f'Contexto: {context}' if context else ''}

            Analise esta consulta e forneça resposta estruturada em JSON.
            Se o usuário está perguntando sobre um valor específico de uma coluna (ex: máximo, mínimo),
            identifique claramente isso no intent_type e nas entidades extraídas.
            """

            response = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
            )

            # Tentar parsear JSON da resposta
            content = response["content"]

            # Extrair JSON da resposta (pode estar em markdown)
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content

            return json.loads(json_str)

        except Exception as e:
            log_error_with_context(e, {"operation": "llm_analysis"})
            # Retornar estrutura básica em caso de erro
            return {
                "confidence_score": 0.5,
                "complexity": "medium",
                "recommendations": []
            }

    def _consolidate_classification(self, user_query: UserQuery, llm_analysis: Dict[str, Any]) -> QueryClassification:
        """Consolida classificação combinando análise local e LLM."""

        # Usar confiança do LLM ou calcular baseado em heurísticas
        confidence_score = llm_analysis.get("confidence_score", 0.7)

        # Consolidar agentes necessários
        required_agents = user_query.analysis_types.copy()
        llm_agents = llm_analysis.get("analysis_types", [])

        for agent_str in llm_agents:
            try:
                agent_type = EDAAnalysisType(agent_str)
                if agent_type not in required_agents:
                    required_agents.append(agent_type)
            except ValueError:
                continue

        # Ordem de processamento (do LLM ou heurística)
        processing_order = []
        if "processing_order" in llm_analysis:
            for agent_str in llm_analysis["processing_order"]:
                try:
                    processing_order.append(EDAAnalysisType(agent_str))
                except ValueError:
                    continue
        else:
            # Ordem padrão heurística
            order_priority = [
                EDAAnalysisType.DESCRIPTIVE,
                EDAAnalysisType.PATTERN,
                EDAAnalysisType.ANOMALY,
                EDAAnalysisType.RELATIONSHIP
            ]
            processing_order = [t for t in order_priority if t in required_agents]

        # Complexidade estimada
        complexity = llm_analysis.get("complexity", "medium")

        return QueryClassification(
            query=user_query,
            confidence_score=confidence_score,
            required_agents=required_agents,
            processing_order=processing_order,
            estimated_complexity=complexity,
            recommendations=llm_analysis.get("recommendations", []),
            metadata={
                "llm_analysis": llm_analysis,
                "processing_time": datetime.now().isoformat()
            }
        )

    def suggest_follow_up_questions(
        self,
        query_classification: QueryClassification,
        csv_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Sugere perguntas de acompanhamento baseadas na análise."""
        suggestions = []
        intent = query_classification.query.intent_type

        follow_up_templates = {
            QueryIntentType.DATA_OVERVIEW: [
                "Gostaria de ver a correlação entre as variáveis numéricas?",
                "Quer identificar outliers nos dados?",
                "Deseja analisar a distribuição de alguma coluna específica?"
            ],
            QueryIntentType.STATISTICAL_SUMMARY: [
                "Gostaria de comparar estas estatísticas entre grupos?",
                "Quer ver a evolução temporal dessas métricas?",
                "Deseja identificar valores atípicos?"
            ],
            QueryIntentType.CORRELATION_ANALYSIS: [
                "Quer explorar relações não-lineares?",
                "Gostaria de ver um mapa de calor das correlações?",
                "Deseja analisar correlações condicionais?"
            ]
        }

        if intent and intent in follow_up_templates:
            suggestions.extend(follow_up_templates[intent])

        return suggestions[:3]  # Limitar a 3 sugestões


# Instância singleton
_query_interpreter: Optional[QueryInterpreter] = None


def get_query_interpreter() -> QueryInterpreter:
    """Obtém instância singleton do QueryInterpreter."""
    global _query_interpreter
    if _query_interpreter is None:
        _query_interpreter = QueryInterpreter()
    return _query_interpreter