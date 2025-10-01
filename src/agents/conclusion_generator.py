"""
Agente especializado na síntese de insights e conclusões finais.
Consolida descobertas de todos os agentes, gera resumos executivos
e mantém histórico de conclusões para referência futura.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.config import get_config
from ..core.llm_manager import get_llm_manager
from ..core.memory_manager import get_memory_manager
from ..core.logger import get_logger, log_analysis_step, log_error_with_context
from ..models.analysis_result import (
    DescriptiveAnalysisResult,
    PatternAnalysisResult,
    AnomalyAnalysisResult,
    RelationshipAnalysisResult
)
from ..models.graph_schema import ConsolidatedResults, DataInsight
from ..models.enums import EDAAnalysisType, ProcessingStatus


class ConclusionGenerationError(Exception):
    """Erro específico para geração de conclusões."""
    pass


class ConclusionGeneratorAgent:
    """Agente especializado em síntese de insights e conclusões finais."""

    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        self.memory_manager = get_memory_manager()
        self.logger = get_logger("conclusion_generator_agent")

    def consolidate_results(
        self,
        descriptive_result: Optional[DescriptiveAnalysisResult] = None,
        pattern_result: Optional[PatternAnalysisResult] = None,
        anomaly_result: Optional[AnomalyAnalysisResult] = None,
        relationship_result: Optional[RelationshipAnalysisResult] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ConsolidatedResults:
        """
        Consolida resultados de todos os agentes em conclusões finais.

        Args:
            descriptive_result: Resultado da análise descritiva
            pattern_result: Resultado da análise de padrões
            anomaly_result: Resultado da análise de anomalias
            relationship_result: Resultado da análise de relacionamentos
            session_id: ID da sessão para memória
            context: Contexto adicional da consulta

        Returns:
            ConsolidatedResults com síntese completa
        """
        start_time = time.time()

        try:
            log_analysis_step("conclusion_generation", "started", {
                "has_descriptive": descriptive_result is not None,
                "has_pattern": pattern_result is not None,
                "has_anomaly": anomaly_result is not None,
                "has_relationship": relationship_result is not None
            })

            # Extrair query_text do contexto
            query_text = context.get("query", "Análise exploratória de dados") if context else "Análise exploratória de dados"

            # Criar resultado consolidado com campos obrigatórios
            consolidated = ConsolidatedResults(
                query_text=query_text,
                response_text=""  # Será preenchido após geração dos insights
            )

            # 1. Coletar todos os insights individuais
            all_insights = self._collect_all_insights(
                descriptive_result, pattern_result, anomaly_result, relationship_result
            )

            # 2. Gerar insights consolidados
            consolidated_insights = self._generate_consolidated_insights(
                all_insights, context, session_id
            )

            # Atribuir insights ao modelo (usar all_insights em vez de consolidated_insights)
            consolidated.all_insights = consolidated_insights
            # Filtrar insights prioritários (alta importância)
            consolidated.priority_insights = [
                insight for insight in consolidated_insights
                if insight.importance == "high"
            ]

            # 3. Gerar resumo textual
            response_text = self._generate_response_text(
                consolidated_insights, descriptive_result, pattern_result,
                anomaly_result, relationship_result, context
            )
            consolidated.response_text = response_text

            # 4. Preencher metadata
            consolidated.metadata = {
                "session_id": session_id,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "insights_count": len(consolidated_insights),
                "priority_insights_count": len(consolidated.priority_insights)
            }

            # 5. Salvar na memória se session_id fornecido
            if session_id:
                self._save_to_memory(session_id, consolidated_insights)

            log_analysis_step(
                "conclusion_generation", "completed",
                {
                    "processing_time": consolidated.metadata["processing_time"],
                    "insights_count": len(consolidated_insights),
                    "priority_insights_count": len(consolidated.priority_insights)
                }
            )

            return consolidated

        except Exception as e:
            log_error_with_context(e, {"operation": "conclusion_generation", "session_id": session_id})
            raise ConclusionGenerationError(f"Erro na geração de conclusões: {e}")

    def _collect_all_insights(
        self,
        descriptive_result: Optional[DescriptiveAnalysisResult],
        pattern_result: Optional[PatternAnalysisResult],
        anomaly_result: Optional[AnomalyAnalysisResult],
        relationship_result: Optional[RelationshipAnalysisResult]
    ) -> Dict[str, List[str]]:
        """Coleta todos os insights dos agentes individuais."""
        all_insights = {
            "descriptive": [],
            "pattern": [],
            "anomaly": [],
            "relationship": []
        }

        try:
            if descriptive_result and descriptive_result.insights:
                all_insights["descriptive"] = descriptive_result.insights

            if pattern_result and pattern_result.insights:
                all_insights["pattern"] = pattern_result.insights

            if anomaly_result and anomaly_result.insights:
                all_insights["anomaly"] = anomaly_result.insights

            if relationship_result and relationship_result.insights:
                all_insights["relationship"] = relationship_result.insights

            return all_insights

        except Exception as e:
            self.logger.error(f"Erro ao coletar insights: {e}")
            return all_insights

    def _generate_consolidated_insights(
        self,
        all_insights: Dict[str, List[str]],
        context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> List[DataInsight]:
        """Gera insights consolidados usando LLM."""
        try:
            # Obter contexto da sessão se disponível
            session_context = ""
            if session_id:
                session_info = self.memory_manager.get_session_context(session_id)
                if session_info:
                    session_context = f"""
                    Contexto da sessão:
                    - Consultas anteriores: {len(session_info.get('recent_queries', []))}
                    - Insights anteriores: {len(session_info.get('high_confidence_insights', []))}
                    - Padrões de análise: {session_info.get('analysis_patterns', {})}
                    """

            # Preparar resumo para o LLM
            insights_summary = "Insights coletados dos agentes especializados:\n\n"

            for analysis_type, insights in all_insights.items():
                if insights:
                    insights_summary += f"**{analysis_type.title()}:**\n"
                    for insight in insights:
                        insights_summary += f"- {insight}\n"
                    insights_summary += "\n"

            # Extrair query original do contexto
            original_query = context.get('query', '') if context else ''

            # Prompt para consolidação
            system_prompt = """
            Você é um especialista em análise exploratória de dados responsável por consolidar insights.
            Sua tarefa PRINCIPAL é responder ESPECIFICAMENTE à pergunta original do usuário.

            IMPORTANTE - ORDEM DE PRIORIDADE:
            1. Responda DIRETAMENTE à pergunta do usuário se ela for específica
            2. Se a pergunta for sobre um valor específico (máximo, mínimo, média, etc.), forneça esse valor PRIMEIRO
            3. Depois, sintetize insights complementares relevantes
            4. NÃO forneça apenas uma visão geral genérica se a pergunta for específica

            Regras para consolidação:
            1. Priorize responder à pergunta EXATA do usuário
            2. Identifique padrões e conexões entre diferentes tipos de análise
            3. Priorize insights acionáveis e práticos
            4. Elimine redundâncias e contradições
            5. Mantenha linguagem clara e objetiva

            Gere entre 1-5 insights RELEVANTES à pergunta do usuário.
            Cada insight deve ser conciso (máximo 200 caracteres) e específico.
            """

            user_prompt = f"""
            PERGUNTA ORIGINAL DO USUÁRIO: "{original_query}"

            {insights_summary}

            {session_context}

            IMPORTANTE: Responda ESPECIFICAMENTE à pergunta do usuário acima.
            Se a pergunta for sobre um valor específico (máximo, mínimo, etc.), inclua esse valor explicitamente nos insights.
            Não forneça apenas uma visão geral genérica se a pergunta for específica.
            """

            response = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
            )

            # Processar resposta em DataInsights
            consolidated_insights = self._parse_insights_from_llm(response["content"])

            return consolidated_insights

        except Exception as e:
            self.logger.warning(f"Erro ao gerar insights consolidados com LLM: {e}")
            # Fallback para consolidação básica
            return self._generate_basic_consolidated_insights(all_insights)

    def _parse_insights_from_llm(self, llm_response: str) -> List[DataInsight]:
        """Converte resposta do LLM em objetos DataInsight."""
        insights = []

        try:
            # Extrair linhas que parecem insights
            lines = llm_response.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:
                    # Remover marcadores
                    clean_line = line.lstrip("- ").lstrip("• ").lstrip("* ").lstrip("1. ").lstrip("2. ").lstrip("3. ")

                    if len(clean_line) > 20 and len(clean_line) < 200:
                        # Calcular confiança baseada no comprimento e especificidade
                        confidence = min(0.9, 0.5 + (len(clean_line) / 300))

                        # Determinar importância baseada em palavras-chave
                        importance = self._calculate_insight_importance(clean_line)

                        insight = DataInsight(
                            insight_text=clean_line,
                            confidence=confidence,
                            analysis_type=EDAAnalysisType.DESCRIPTIVE,  # Padrão
                            importance=importance,
                            timestamp=datetime.now(),
                            metadata={"source": "llm_consolidation"}
                        )
                        insights.append(insight)

            return insights[:7]  # Máximo 7 insights

        except Exception as e:
            self.logger.warning(f"Erro ao processar insights do LLM: {e}")
            return []

    def _calculate_insight_importance(self, insight_text: str) -> str:
        """Calcula importância de um insight baseado em palavras-chave."""
        try:
            text_lower = insight_text.lower()

            # Palavras que indicam alta importância
            high_importance_words = [
                "crítico", "importante", "significativo", "forte", "alto", "problema",
                "oportunidade", "recomenda", "atenção", "cuidado", "risco"
            ]

            # Palavras que indicam importância moderada
            moderate_importance_words = [
                "moderado", "tendência", "padrão", "correlação", "sugere", "indica"
            ]

            high_score = sum(1 for word in high_importance_words if word in text_lower)
            moderate_score = sum(1 for word in moderate_importance_words if word in text_lower)

            # Determinar nível de importância
            if high_score >= 2:
                return "high"
            elif high_score >= 1 or moderate_score >= 2:
                return "medium"
            else:
                return "low"

        except Exception as e:
            self.logger.warning(f"Erro ao calcular importância: {e}")
            return "medium"  # Importância padrão

    def _generate_basic_consolidated_insights(self, all_insights: Dict[str, List[str]]) -> List[DataInsight]:
        """Gera insights consolidados básicos como fallback."""
        consolidated = []

        try:
            # Resumir por categoria
            for analysis_type, insights_list in all_insights.items():
                if insights_list:
                    # Pegar o primeiro insight como representativo
                    representative_insight = insights_list[0]

                    insight = DataInsight(
                        insight_text=f"[{analysis_type.title()}] {representative_insight}",
                        confidence=0.7,
                        analysis_type=EDAAnalysisType(analysis_type.upper()) if analysis_type.upper() in EDAAnalysisType.__members__ else EDAAnalysisType.DESCRIPTIVE,
                        importance=0.6,
                        timestamp=datetime.now(),
                        metadata={"source": "basic_consolidation", "total_insights": len(insights_list)}
                    )
                    consolidated.append(insight)

            # Adicionar insight geral se há múltiplas análises
            if len([insights for insights in all_insights.values() if insights]) > 1:
                general_insight = DataInsight(
                    insight_text="Análise EDA completa realizada com múltiplos métodos de investigação",
                    confidence=0.8,
                    analysis_type=EDAAnalysisType.DESCRIPTIVE,
                    importance=0.7,
                    timestamp=datetime.now(),
                    metadata={"source": "basic_consolidation", "type": "summary"}
                )
                consolidated.append(general_insight)

            return consolidated

        except Exception as e:
            self.logger.error(f"Erro na consolidação básica: {e}")
            return []

    def _generate_executive_summary(
        self,
        descriptive_result: Optional[DescriptiveAnalysisResult],
        pattern_result: Optional[PatternAnalysisResult],
        anomaly_result: Optional[AnomalyAnalysisResult],
        relationship_result: Optional[RelationshipAnalysisResult],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Gera resumo executivo das análises."""
        try:
            # Coletar estatísticas principais
            summary_parts = []

            # Informações do dataset
            if descriptive_result and descriptive_result.dataset_overview:
                overview = descriptive_result.dataset_overview
                summary_parts.append(
                    f"Dataset analisado: {overview.get('total_rows', 0)} linhas, "
                    f"{overview.get('total_columns', 0)} colunas, "
                    f"{overview.get('completeness_percentage', 0):.1f}% completo"
                )

            # Padrões identificados
            if pattern_result and pattern_result.trends:
                summary_parts.append(f"Identificadas {len(pattern_result.trends)} tendências nos dados")

            # Anomalias detectadas
            if anomaly_result and anomaly_result.outlier_statistics:
                outliers_count = anomaly_result.outlier_statistics.get('total_outliers_detected', 0)
                if outliers_count > 0:
                    summary_parts.append(f"Detectados {outliers_count} outliers")

            # Relacionamentos encontrados
            if relationship_result and relationship_result.strongest_correlations:
                strong_corrs = len([c for c in relationship_result.strongest_correlations if c['strength'] in ['strong', 'very_strong']])
                if strong_corrs > 0:
                    summary_parts.append(f"{strong_corrs} correlações fortes identificadas")

            # Consolidar em resumo
            if summary_parts:
                executive_summary = f"""
                Resumo Executivo da Análise EDA:

                {'; '.join(summary_parts)}.

                As análises revelaram características importantes do dataset que podem orientar
                decisões de negócio e próximas etapas de investigação.
                """.strip()
            else:
                executive_summary = "Análise exploratória de dados concluída com sucesso."

            return executive_summary

        except Exception as e:
            self.logger.error(f"Erro ao gerar resumo executivo: {e}")
            return "Análise EDA realizada. Consulte os insights individuais para detalhes."

    def _generate_final_recommendations(
        self,
        descriptive_result: Optional[DescriptiveAnalysisResult],
        pattern_result: Optional[PatternAnalysisResult],
        anomaly_result: Optional[AnomalyAnalysisResult],
        relationship_result: Optional[RelationshipAnalysisResult]
    ) -> List[str]:
        """Gera recomendações finais baseadas em todas as análises."""
        recommendations = []

        try:
            # Recomendações de qualidade de dados
            if descriptive_result and descriptive_result.dataset_overview:
                completeness = descriptive_result.dataset_overview.get('completeness_percentage', 100)
                if completeness < 95:
                    recommendations.append("Investigar e tratar dados ausentes para melhorar qualidade do dataset")

                duplicates = descriptive_result.dataset_overview.get('duplicate_percentage', 0)
                if duplicates > 5:
                    recommendations.append("Revisar e remover linhas duplicadas identificadas")

            # Recomendações de anomalias
            if anomaly_result and anomaly_result.recommendations:
                recommendations.extend(anomaly_result.recommendations[:2])  # Top 2

            # Recomendações de relacionamentos
            if relationship_result and relationship_result.multicollinearity_analysis:
                high_corr_pairs = relationship_result.multicollinearity_analysis.get('high_correlation_pairs', [])
                if high_corr_pairs:
                    recommendations.append("Considerar análise de multicolinearidade antes de modelagem preditiva")

            # Recomendações de análises adicionais
            recommendations.append("Considerar análises de visualização para melhor compreensão dos padrões")

            if pattern_result and pattern_result.temporal_patterns:
                recommendations.append("Explorar análises temporais detalhadas se relevante para o negócio")

            return recommendations[:6]  # Máximo 6 recomendações

        except Exception as e:
            self.logger.error(f"Erro ao gerar recomendações finais: {e}")
            return ["Revisar resultados individuais de cada análise para próximos passos"]

    def _save_to_memory(self, session_id: str, insights: List[DataInsight]) -> None:
        """Salva insights consolidados na memória da sessão."""
        try:
            if insights:
                self.memory_manager.save_insights(session_id, insights)
                self.logger.info(f"Salvos {len(insights)} insights consolidados na memória da sessão {session_id}")

        except Exception as e:
            self.logger.warning(f"Erro ao salvar insights na memória: {e}")


    def _generate_response_text(
        self,
        insights: List[DataInsight],
        descriptive_result: Optional[DescriptiveAnalysisResult],
        pattern_result: Optional[PatternAnalysisResult],
        anomaly_result: Optional[AnomalyAnalysisResult],
        relationship_result: Optional[RelationshipAnalysisResult],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Gera texto de resposta consolidado baseado nos insights e na query específica.

        Args:
            insights: Lista de insights gerados
            descriptive_result: Resultado da análise descritiva
            pattern_result: Resultado da análise de padrões
            anomaly_result: Resultado da análise de anomalias
            relationship_result: Resultado da análise de relacionamentos
            context: Contexto da consulta

        Returns:
            Texto de resposta consolidado e contextualizado
        """
        try:
            # Extrair query do contexto
            user_query = context.get('query', '') if context else ''
            query_lower = user_query.lower()

            response_parts = []

            # PRIORIDADE 1: Responder à query específica primeiro
            # Verificar se é uma query sobre valores específicos
            is_specific_value_query = any(keyword in query_lower for keyword in [
                'máximo', 'max', 'mínimo', 'min', 'média', 'mean', 'soma', 'sum',
                'total', 'qual', 'what', 'valor', 'value'
            ])

            if is_specific_value_query and descriptive_result and descriptive_result.column_summaries:
                # Tentar responder diretamente à query
                specific_answer = self._extract_specific_answer(user_query, descriptive_result)
                if specific_answer:
                    response_parts.append(specific_answer)

            # PRIORIDADE 2: Adicionar insights prioritários (relevantes à query)
            priority_insights = [i for i in insights if i.importance == "high"]
            if priority_insights:
                # Se já respondeu especificamente, adicionar apenas insights complementares
                if not is_specific_value_query or not response_parts:
                    response_parts.append("\nPrincipais descobertas:")
                    for idx, insight in enumerate(priority_insights[:5], 1):
                        response_parts.append(f"{idx}. {insight.insight_text}")
                else:
                    # Adicionar apenas insights relevantes
                    for insight in priority_insights[:2]:
                        response_parts.append(f"\n{insight.insight_text}")

            # PRIORIDADE 3: Outros insights (apenas se ainda não há resposta satisfatória)
            if not response_parts:
                other_insights = [i for i in insights if i.importance != "high"]
                if other_insights:
                    response_parts.append("Observações sobre os dados:")
                    for insight in other_insights[:3]:
                        response_parts.append(f"• {insight.insight_text}")

            # PRIORIDADE 4: Informações contextuais adicionais (apenas se relevante)
            if anomaly_result and anomaly_result.outliers_by_column and 'anomalia' in query_lower or 'outlier' in query_lower:
                total_outliers = sum(
                    len(info.get("outliers", []))
                    for info in anomaly_result.outliers_by_column.values()
                )
                if total_outliers > 0:
                    response_parts.append(
                        f"\nForam detectados {total_outliers} valores atípicos nos dados."
                    )

            # Compilar resposta
            response_text = "\n".join(response_parts)

            # FALLBACK: Se ainda não há resposta, gerar uma básica
            if not response_text.strip():
                if descriptive_result and descriptive_result.dataset_overview:
                    overview = descriptive_result.dataset_overview
                    response_text = (
                        f"Dataset analisado contém {overview.get('total_rows', 0)} linhas e "
                        f"{overview.get('total_columns', 0)} colunas. "
                        f"Completude: {overview.get('completeness_percentage', 0):.1f}%"
                    )
                else:
                    response_text = "Análise concluída. Os dados foram processados."

            return response_text

        except Exception as e:
            self.logger.error(f"Erro ao gerar texto de resposta: {e}")
            return "Análise concluída. Para mais detalhes, consulte os resultados individuais de cada análise."

    def _extract_specific_answer(self, user_query: str, descriptive_result: DescriptiveAnalysisResult) -> Optional[str]:
        """
        Extrai resposta específica para queries sobre valores particulares.

        Args:
            user_query: Query do usuário
            descriptive_result: Resultado da análise descritiva

        Returns:
            Resposta específica ou None se não conseguir extrair
        """
        try:
            query_lower = user_query.lower()

            # Identificar coluna mencionada
            target_column = None
            for summary in descriptive_result.column_summaries:
                if summary.column_name.lower() in query_lower:
                    target_column = summary
                    break

            if not target_column:
                return None

            # Identificar tipo de valor solicitado
            if any(keyword in query_lower for keyword in ['máximo', 'max', 'maximum', 'maior']):
                if target_column.max_value is not None:
                    return f"O valor máximo da coluna '{target_column.column_name}' é {target_column.max_value}."

            elif any(keyword in query_lower for keyword in ['mínimo', 'min', 'minimum', 'menor']):
                if target_column.min_value is not None:
                    return f"O valor mínimo da coluna '{target_column.column_name}' é {target_column.min_value}."

            elif any(keyword in query_lower for keyword in ['média', 'mean', 'average']):
                if target_column.mean is not None:
                    return f"A média da coluna '{target_column.column_name}' é {target_column.mean:.2f}."

            elif any(keyword in query_lower for keyword in ['soma', 'sum', 'total']) and 'linha' not in query_lower:
                if target_column.mean is not None and target_column.count > 0:
                    total = target_column.mean * target_column.count
                    return f"A soma total da coluna '{target_column.column_name}' é aproximadamente {total:.2f}."

            elif any(keyword in query_lower for keyword in ['mediana', 'median']):
                if target_column.median is not None:
                    return f"A mediana da coluna '{target_column.column_name}' é {target_column.median}."

            return None

        except Exception as e:
            self.logger.warning(f"Erro ao extrair resposta específica: {e}")
            return None


# Instância singleton
_conclusion_generator_agent: Optional[ConclusionGeneratorAgent] = None


def get_conclusion_generator_agent() -> ConclusionGeneratorAgent:
    """Obtém instância singleton do ConclusionGeneratorAgent."""
    global _conclusion_generator_agent
    if _conclusion_generator_agent is None:
        _conclusion_generator_agent = ConclusionGeneratorAgent()
    return _conclusion_generator_agent