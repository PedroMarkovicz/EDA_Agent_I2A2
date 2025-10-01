"""
Definição da estrutura de estado compartilhado do LangGraph.
Especifica schema de dados que transita entre nós, incluindo resultados
de análises, contexto da sessão e informações de controle do fluxo.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import pandas as pd

from ..models.query_schema import UserQuery, QueryClassification
from ..models.analysis_result import (
    DescriptiveAnalysisResult,
    PatternAnalysisResult,
    AnomalyAnalysisResult,
    RelationshipAnalysisResult,
    VisualizationResult,
)
from ..models.graph_schema import FinalResponse, MemoryContext
from ..models.enums import EDAAnalysisType


@dataclass
class EDAState:
    """Estado compartilhado do LangGraph para análise EDA."""

    # Dados de entrada
    csv_data: Optional[pd.DataFrame] = None
    csv_metadata: Optional[Dict[str, Any]] = None
    user_query: str = ""
    session_id: Optional[str] = None  # ID da sessão para memória contextual

    # Classificação e roteamento
    query_classification: List[str] = field(default_factory=list)
    required_agents: List[str] = field(default_factory=list)
    processed_query: Optional[UserQuery] = None
    query_classification_result: Optional[QueryClassification] = None

    # Resultados de análise
    descriptive_results: Optional[DescriptiveAnalysisResult] = None
    pattern_results: Optional[PatternAnalysisResult] = None
    anomaly_results: Optional[AnomalyAnalysisResult] = None
    relationship_results: Optional[RelationshipAnalysisResult] = None

    # Código e visualizações
    generated_code: str = ""
    execution_results: Optional[Dict[str, Any]] = None
    visualizations: List[VisualizationResult] = field(default_factory=list)

    # Tool orchestration (novo sistema baseado em ferramentas)
    tool_execution_results: List[Dict[str, Any]] = field(default_factory=list)
    tool_based_response: str = ""

    # Contexto e memória
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    analysis_context: Dict[str, Any] = field(default_factory=dict)
    memory_context: Optional[MemoryContext] = None

    # Resposta final
    final_insights: str = ""
    response_data: Optional[FinalResponse] = None

    # Controle de fluxo
    completed_analyses: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_completed_analysis(self, analysis_type: EDAAnalysisType) -> None:
        """Marca um tipo de análise como completado."""
        self.completed_analyses.add(analysis_type.value)

    def is_analysis_completed(self, analysis_type: EDAAnalysisType) -> bool:
        """Verifica se um tipo de análise foi completado."""
        return analysis_type.value in self.completed_analyses

    def add_error(self, error_message: str) -> None:
        """Adiciona um erro à lista de erros."""
        self.errors.append(error_message)

    def has_errors(self) -> bool:
        """Verifica se há erros registrados."""
        return len(self.errors) > 0

    def get_analysis_result(self, analysis_type: EDAAnalysisType) -> Optional[Any]:
        """Obtém o resultado de um tipo específico de análise."""
        result_mapping = {
            EDAAnalysisType.DESCRIPTIVE: self.descriptive_results,
            EDAAnalysisType.PATTERN: self.pattern_results,
            EDAAnalysisType.ANOMALY: self.anomaly_results,
            EDAAnalysisType.RELATIONSHIP: self.relationship_results,
        }
        return result_mapping.get(analysis_type)

    def set_analysis_result(self, analysis_type: EDAAnalysisType, result: Any) -> None:
        """Define o resultado de um tipo específico de análise."""
        if analysis_type == EDAAnalysisType.DESCRIPTIVE:
            self.descriptive_results = result
        elif analysis_type == EDAAnalysisType.PATTERN:
            self.pattern_results = result
        elif analysis_type == EDAAnalysisType.ANOMALY:
            self.anomaly_results = result
        elif analysis_type == EDAAnalysisType.RELATIONSHIP:
            self.relationship_results = result

    def should_execute_agent(self, agent_name: str) -> bool:
        """Determina se um agente deve ser executado baseado na classificacao da consulta."""
        # Mapeamento de nomes de agentes para tipos de análise
        agent_to_analysis_type = {
            "data_analyzer": EDAAnalysisType.DESCRIPTIVE.value,
            "pattern_detector": EDAAnalysisType.PATTERN.value,
            "anomaly_detector": EDAAnalysisType.ANOMALY.value,
            "relationship_analyzer": EDAAnalysisType.RELATIONSHIP.value,
            "code_generator": "code_generator"  # Código sempre é executado
        }

        analysis_type = agent_to_analysis_type.get(agent_name)
        return analysis_type in self.required_agents if analysis_type else False

    def get_all_results(self) -> Dict[str, Any]:
        """Retorna todos os resultados de analise disponíveis."""
        return {
            "descriptive": self.descriptive_results,
            "pattern": self.pattern_results,
            "anomaly": self.anomaly_results,
            "relationship": self.relationship_results,
            "visualizations": self.visualizations,
            "generated_code": self.generated_code,
            "execution_results": self.execution_results
        }

    def is_ready_for_synthesis(self) -> bool:
        """Verifica se o estado está pronto para síntese de resultados."""
        # Verificar se pelo menos uma análise foi concluída
        has_results = any([
            self.descriptive_results is not None,
            self.pattern_results is not None,
            self.anomaly_results is not None,
            self.relationship_results is not None
        ])
        return has_results and not self.has_errors()

    def reset_for_new_query(self) -> None:
        """Reseta o estado para uma nova consulta mantendo contexto de sessão."""
        # Limpar resultados da consulta anterior
        self.descriptive_results = None
        self.pattern_results = None
        self.anomaly_results = None
        self.relationship_results = None
        self.generated_code = ""
        self.execution_results = None
        self.visualizations = []
        self.final_insights = ""
        self.response_data = None
        self.completed_analyses = set()
        self.errors = []
        self.query_classification = []
        self.required_agents = []

        # Manter csv_data, conversation_history e analysis_context