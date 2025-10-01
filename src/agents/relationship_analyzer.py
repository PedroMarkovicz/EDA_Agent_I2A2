"""
Agente focado na análise de relacionamentos e correlações entre variáveis.
Calcula diversos tipos de correlação, identifica dependências não-lineares
e detecta problemas de multicolinearidade nos datasets.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ..core.config import get_config
from ..core.llm_manager import get_llm_manager
from ..core.logger import get_logger, log_analysis_step, log_error_with_context
from ..models.analysis_result import RelationshipAnalysisResult, VisualizationResult
from ..models.enums import EDAAnalysisType, ProcessingStatus, VisualizationType


class RelationshipAnalysisError(Exception):
    """Erro específico para análise de relacionamentos."""
    pass


class RelationshipAnalyzerAgent:
    """Agente especializado em análise de correlações e relacionamentos."""

    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        self.logger = get_logger("relationship_analyzer_agent")

    def analyze(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> RelationshipAnalysisResult:
        """
        Executa análise de relacionamentos e correlações entre variáveis.

        Args:
            data: DataFrame para análise
            context: Contexto adicional da consulta

        Returns:
            RelationshipAnalysisResult com análise completa
        """
        start_time = time.time()

        try:
            log_analysis_step("relationship_analysis", "started", {"shape": data.shape})

            result = RelationshipAnalysisResult(
                analysis_type=EDAAnalysisType.RELATIONSHIP,
                status=ProcessingStatus.IN_PROGRESS
            )

            # 1. Análise de correlações numéricas
            result.correlation_matrix = self._calculate_correlation_matrix(data)

            # 2. Correlações categóricas
            result.categorical_associations = self._analyze_categorical_associations(data)

            # 3. Análise de multicolinearidade
            result.multicollinearity_analysis = self._analyze_multicollinearity(data)

            # 4. Relacionamentos não-lineares
            result.nonlinear_relationships = self._detect_nonlinear_relationships(data)

            # 5. Correlações mais fortes
            result.strongest_correlations = self._identify_strongest_correlations(result.correlation_matrix)

            # 6. Gerar insights usando LLM
            insights = self._generate_insights(data, result, context)
            result.insights = insights

            # 7. Marcar como concluído
            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - start_time

            log_analysis_step(
                "relationship_analysis", "completed",
                {
                    "processing_time": result.processing_time,
                    "insights_count": len(result.insights),
                    "correlations_analyzed": len(result.correlation_matrix) if result.correlation_matrix else 0,
                    "strong_correlations_found": len(result.strongest_correlations)
                }
            )

            return result

        except Exception as e:
            log_error_with_context(e, {"operation": "relationship_analysis", "data_shape": data.shape})
            raise RelationshipAnalysisError(f"Erro na análise de relacionamentos: {e}")

    def _calculate_correlation_matrix(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula matriz de correlação usando múltiplos métodos."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) < 2:
                return {"error": "Menos de 2 colunas numéricas para análise de correlação"}

            # Remover colunas com variância zero
            numeric_data = numeric_data.loc[:, numeric_data.var() != 0]

            if len(numeric_data.columns) < 2:
                return {"error": "Dados insuficientes após remoção de colunas constantes"}

            correlation_results = {
                "methods": ["pearson", "spearman", "kendall"],
                "correlations": {},
                "summary_statistics": {}
            }

            # Calcular correlações por método
            for method in correlation_results["methods"]:
                try:
                    if method == "pearson":
                        corr_matrix = numeric_data.corr(method='pearson')
                    elif method == "spearman":
                        corr_matrix = numeric_data.corr(method='spearman')
                    elif method == "kendall":
                        corr_matrix = numeric_data.corr(method='kendall')

                    # Converter para dicionário serializável
                    correlation_results["correlations"][method] = corr_matrix.fillna(0).to_dict()

                    # Estatísticas da matriz
                    correlation_results["summary_statistics"][method] = self._calculate_correlation_statistics(corr_matrix)

                except Exception as e:
                    self.logger.warning(f"Erro ao calcular correlação {method}: {e}")
                    correlation_results["correlations"][method] = {}

            return correlation_results

        except Exception as e:
            self.logger.error(f"Erro ao calcular matriz de correlação: {e}")
            return {"error": str(e)}

    def _calculate_correlation_statistics(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas resumidas da matriz de correlação."""
        try:
            # Obter valores da matriz triangular superior (excluindo diagonal)
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            upper_triangle = corr_matrix.where(mask)
            values = upper_triangle.stack().dropna()

            if len(values) == 0:
                return {}

            stats = {
                "mean_correlation": float(values.mean()),
                "median_correlation": float(values.median()),
                "std_correlation": float(values.std()),
                "max_correlation": float(values.max()),
                "min_correlation": float(values.min()),
                "high_correlations_count": int((abs(values) > 0.7).sum()),
                "moderate_correlations_count": int(((abs(values) > 0.3) & (abs(values) <= 0.7)).sum()),
                "weak_correlations_count": int((abs(values) <= 0.3).sum())
            }

            return stats

        except Exception as e:
            self.logger.warning(f"Erro ao calcular estatísticas de correlação: {e}")
            return {}

    def _analyze_categorical_associations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa associações entre variáveis categóricas."""
        try:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns

            if len(categorical_cols) < 2:
                return {"message": "Menos de 2 colunas categóricas para análise de associação"}

            associations = {
                "categorical_pairs": [],
                "cross_tabulations": {},
                "association_strength": {}
            }

            # Analisar pares de variáveis categóricas (limitar a primeiras 5 colunas)
            for i, col1 in enumerate(categorical_cols[:5]):
                for col2 in categorical_cols[i+1:6]:  # Evitar analisar com si mesmo
                    try:
                        pair_name = f"{col1}_vs_{col2}"
                        associations["categorical_pairs"].append(pair_name)

                        # Tabela de contingência
                        contingency_table = pd.crosstab(data[col1], data[col2])
                        associations["cross_tabulations"][pair_name] = contingency_table.to_dict()

                        # Força da associação usando Cramér's V simplificado
                        association_strength = self._calculate_cramers_v_simplified(contingency_table)
                        associations["association_strength"][pair_name] = association_strength

                    except Exception as e:
                        self.logger.warning(f"Erro ao analisar associação {col1} vs {col2}: {e}")

            return associations

        except Exception as e:
            self.logger.error(f"Erro na análise de associações categóricas: {e}")
            return {"error": str(e)}

    def _calculate_cramers_v_simplified(self, contingency_table: pd.DataFrame) -> float:
        """Calcula uma versão simplificada do Cramér's V."""
        try:
            # Implementação simplificada baseada na variação dos valores
            total = contingency_table.sum().sum()
            if total == 0:
                return 0.0

            # Calcular a variação normalizada
            expected_uniform = total / (contingency_table.shape[0] * contingency_table.shape[1])
            observed_values = contingency_table.values.flatten()
            expected_values = np.full_like(observed_values, expected_uniform)

            # Calcular uma medida de desvio normalizada
            if expected_uniform > 0:
                deviations = np.abs(observed_values - expected_values) / expected_uniform
                return min(np.mean(deviations), 1.0)
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Erro ao calcular Cramér's V: {e}")
            return 0.0

    def _analyze_multicollinearity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analisa multicolinearidade entre variáveis numéricas."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) < 2:
                return {"message": "Dados insuficientes para análise de multicolinearidade"}

            # Remover colunas com variância zero
            numeric_data = numeric_data.loc[:, numeric_data.var() != 0]

            if len(numeric_data.columns) < 2:
                return {"message": "Dados insuficientes após remoção de colunas constantes"}

            multicollinearity = {
                "high_correlation_pairs": [],
                "vif_approximation": {},
                "condition_index": None
            }

            # Identificar pares com alta correlação
            corr_matrix = numeric_data.corr(method='pearson')
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Evitar duplicatas
                        correlation = corr_matrix.loc[col1, col2]
                        if abs(correlation) > 0.8:  # Limiar para alta correlação
                            multicollinearity["high_correlation_pairs"].append({
                                "variable_1": col1,
                                "variable_2": col2,
                                "correlation": float(correlation),
                                "severity": "high" if abs(correlation) > 0.9 else "moderate"
                            })

            # Aproximação simplificada do VIF usando correlação
            for col in numeric_data.columns:
                try:
                    other_cols = [c for c in numeric_data.columns if c != col]
                    if len(other_cols) > 0:
                        # Correlação média com outras variáveis
                        correlations = [abs(corr_matrix.loc[col, other_col]) for other_col in other_cols]
                        max_correlation = max(correlations) if correlations else 0

                        # Aproximação simples do VIF
                        vif_approx = 1 / (1 - max_correlation**2) if max_correlation < 0.99 else float('inf')
                        multicollinearity["vif_approximation"][col] = {
                            "vif_estimate": float(vif_approx) if vif_approx != float('inf') else 999.0,
                            "max_correlation": float(max_correlation),
                            "multicollinearity_level": self._classify_multicollinearity(vif_approx)
                        }

                except Exception as e:
                    self.logger.warning(f"Erro ao calcular VIF para {col}: {e}")

            return multicollinearity

        except Exception as e:
            self.logger.error(f"Erro na análise de multicolinearidade: {e}")
            return {"error": str(e)}

    def _classify_multicollinearity(self, vif: float) -> str:
        """Classifica o nível de multicolinearidade baseado no VIF."""
        if vif >= 10:
            return "high"
        elif vif >= 5:
            return "moderate"
        elif vif >= 2.5:
            return "low"
        else:
            return "minimal"

    def _detect_nonlinear_relationships(self, data: pd.DataFrame) -> List[str]:
        """Detecta potenciais relacionamentos não-lineares."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            nonlinear_relationships = []

            if len(numeric_data.columns) < 2:
                return []

            # Analisar pares de variáveis numéricas (limitar para performance)
            columns = numeric_data.columns[:6]  # Máximo 6 colunas

            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    try:
                        data1 = numeric_data[col1].dropna()
                        data2 = numeric_data[col2].dropna()

                        # Encontrar índices comuns
                        common_idx = data1.index.intersection(data2.index)
                        if len(common_idx) < 10:  # Dados insuficientes
                            continue

                        x = data1.loc[common_idx]
                        y = data2.loc[common_idx]

                        # Correlação linear
                        linear_corr = x.corr(y)

                        # Detectar não-linearidade usando correlação de ranks vs linear
                        spearman_corr = x.corr(y, method='spearman')

                        # Se correlação de Spearman é significativamente maior que Pearson
                        if abs(spearman_corr) - abs(linear_corr) > 0.2 and abs(spearman_corr) > 0.5:
                            nonlinear_relationships.append(
                                f"Relacionamento não-linear detectado entre '{col1}' e '{col2}' "
                                f"(Spearman: {spearman_corr:.3f}, Pearson: {linear_corr:.3f})"
                            )

                        # Detectar relacionamentos em U ou forma de parábola
                        if len(x) > 20:
                            # Correlação com quadrado da variável
                            x_squared_corr = (x**2).corr(y)
                            if abs(x_squared_corr) > abs(linear_corr) + 0.2 and abs(x_squared_corr) > 0.5:
                                nonlinear_relationships.append(
                                    f"Relacionamento quadrático detectado entre '{col1}' e '{col2}' "
                                    f"(R² > R linear)"
                                )

                    except Exception as e:
                        self.logger.warning(f"Erro ao analisar relacionamento não-linear {col1} vs {col2}: {e}")

            return nonlinear_relationships[:10]  # Limitar a 10 relacionamentos

        except Exception as e:
            self.logger.error(f"Erro na detecção de relacionamentos não-lineares: {e}")
            return []

    def _identify_strongest_correlations(self, correlation_matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica as correlações mais fortes (positivas e negativas)."""
        try:
            if "correlations" not in correlation_matrix or "pearson" not in correlation_matrix["correlations"]:
                return []

            pearson_corr = correlation_matrix["correlations"]["pearson"]
            strong_correlations = []

            # Converter de volta para DataFrame para facilitar análise
            corr_df = pd.DataFrame(pearson_corr)

            for i, col1 in enumerate(corr_df.columns):
                for j, col2 in enumerate(corr_df.columns):
                    if i < j:  # Evitar duplicatas e diagonal
                        correlation_value = corr_df.loc[col1, col2]

                        if not np.isnan(correlation_value) and abs(correlation_value) > 0.3:  # Limiar mínimo
                            strong_correlations.append({
                                "variable_1": col1,
                                "variable_2": col2,
                                "correlation": float(correlation_value),
                                "strength": self._classify_correlation_strength(correlation_value),
                                "direction": "positive" if correlation_value > 0 else "negative"
                            })

            # Ordenar por força absoluta da correlação
            strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            return strong_correlations[:15]  # Top 15 correlações

        except Exception as e:
            self.logger.error(f"Erro ao identificar correlações mais fortes: {e}")
            return []

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classifica a força da correlação."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"

    def _generate_insights(self, data: pd.DataFrame, result: RelationshipAnalysisResult, context: Optional[Dict[str, Any]]) -> List[str]:
        """Gera insights usando LLM baseado na análise de relacionamentos."""
        try:
            # Preparar resumo para o LLM
            relationship_summary = f"""
            Análise de relacionamentos em dataset com {len(data)} linhas e {len(data.columns)} colunas.

            Correlações encontradas:
            - Correlações fortes: {len([c for c in result.strongest_correlations if c['strength'] in ['strong', 'very_strong']])}
            - Correlações moderadas: {len([c for c in result.strongest_correlations if c['strength'] == 'moderate'])}

            Multicolinearidade:
            - Pares com alta correlação: {len(result.multicollinearity_analysis.get('high_correlation_pairs', []))}

            Relacionamentos não-lineares detectados: {len(result.nonlinear_relationships)}

            Principais correlações:
            """

            # Adicionar top 5 correlações
            for i, corr in enumerate(result.strongest_correlations[:5]):
                relationship_summary += f"\n- {corr['variable_1']} ↔ {corr['variable_2']}: {corr['correlation']:.3f} ({corr['strength']})"

            # Prompt para o LLM
            system_prompt = """
            Você é um especialista em análise de relacionamentos entre variáveis.
            Analise os resultados das correlações e gere insights práticos sobre:
            1. Significado dos relacionamentos encontrados
            2. Implicações da multicolinearidade detectada
            3. Relevância dos relacionamentos não-lineares
            4. Sugestões para análises adicionais

            Forneça entre 3-5 insights concisos e acionáveis.
            """

            user_prompt = f"""
            Análise de relacionamentos:
            {relationship_summary}

            {f"Contexto da consulta: {context.get('query_text', '')}" if context else ""}

            Gere insights sobre os relacionamentos encontrados.
            """

            response = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
            )

            # Processar resposta do LLM
            insights_text = response["content"]
            insights = [
                line.strip().lstrip("- ").lstrip("• ").lstrip("* ")
                for line in insights_text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            # Filtrar insights válidos
            valid_insights = [insight for insight in insights if len(insight) > 20 and len(insight) < 200]

            return valid_insights[:5] if valid_insights else ["Análise de relacionamentos concluída."]

        except Exception as e:
            self.logger.warning(f"Erro ao gerar insights com LLM: {e}")
            # Fallback para insights básicos
            return self._generate_basic_insights(result)

    def _generate_basic_insights(self, result: RelationshipAnalysisResult) -> List[str]:
        """Gera insights básicos sem LLM como fallback."""
        insights = []

        # Insights sobre correlações
        strong_correlations = len([c for c in result.strongest_correlations if c['strength'] in ['strong', 'very_strong']])
        if strong_correlations > 0:
            insights.append(f"Identificadas {strong_correlations} correlações fortes entre variáveis.")
        else:
            insights.append("Correlações fracas ou moderadas entre a maioria das variáveis.")

        # Insights sobre multicolinearidade
        high_corr_pairs = len(result.multicollinearity_analysis.get('high_correlation_pairs', []))
        if high_corr_pairs > 0:
            insights.append(f"Detectados {high_corr_pairs} pares de variáveis com alta correlação (potencial multicolinearidade).")

        # Insights sobre relacionamentos não-lineares
        if result.nonlinear_relationships:
            insights.append(f"Identificados {len(result.nonlinear_relationships)} possíveis relacionamentos não-lineares.")

        return insights[:3]


# Instância singleton
_relationship_analyzer_agent: Optional[RelationshipAnalyzerAgent] = None


def get_relationship_analyzer_agent() -> RelationshipAnalyzerAgent:
    """Obtém instância singleton do RelationshipAnalyzerAgent."""
    global _relationship_analyzer_agent
    if _relationship_analyzer_agent is None:
        _relationship_analyzer_agent = RelationshipAnalyzerAgent()
    return _relationship_analyzer_agent