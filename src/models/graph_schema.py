"""
Schema para representação de conclusões finais e insights consolidados.
Define modelos para estruturar descobertas finais, recomendações
e resumos executivos gerados pelo sistema de análise EDA.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from .enums import EDAAnalysisType, ProcessingStatus
from .analysis_result import VisualizationResult


class DataInsight(BaseModel):
    """Insight individual sobre os dados."""

    insight_text: str = Field(description="Texto do insight")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiança no insight")
    analysis_type: EDAAnalysisType = Field(description="Tipo de análise que gerou o insight")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Dados que suportam o insight")
    importance: str = Field(description="Importância do insight (high, medium, low)")


class ExecutiveSummary(BaseModel):
    """Resumo executivo da análise completa."""

    dataset_name: Optional[str] = Field(None, description="Nome do dataset analisado")
    total_rows: int = Field(description="Total de linhas no dataset")
    total_columns: int = Field(description="Total de colunas no dataset")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da análise")

    key_findings: List[str] = Field(default_factory=list, description="Principais descobertas")
    data_quality_assessment: Dict[str, Any] = Field(default_factory=dict, description="Avaliação da qualidade dos dados")
    recommended_next_steps: List[str] = Field(default_factory=list, description="Próximos passos recomendados")

    analysis_completeness: Dict[EDAAnalysisType, bool] = Field(default_factory=dict, description="Completude por tipo de análise")
    processing_summary: Dict[str, Any] = Field(default_factory=dict, description="Resumo do processamento")


class ConsolidatedResults(BaseModel):
    """Resultados consolidados de todas as análises."""

    query_text: str = Field(description="Consulta original do usuário")
    response_text: str = Field(description="Resposta textual consolidada")

    all_insights: List[DataInsight] = Field(default_factory=list, description="Todos os insights gerados")
    priority_insights: List[DataInsight] = Field(default_factory=list, description="Insights de alta prioridade")

    visualizations_summary: List[VisualizationResult] = Field(default_factory=list, description="Resumo das visualizações")

    analysis_coverage: Dict[EDAAnalysisType, ProcessingStatus] = Field(default_factory=dict, description="Cobertura por tipo de análise")
    confidence_scores: Dict[EDAAnalysisType, float] = Field(default_factory=dict, description="Scores de confiança por análise")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados da resposta")


class MemoryContext(BaseModel):
    """Contexto de memória para análises anteriores."""

    session_id: str = Field(description="ID da sessão")
    user_id: Optional[str] = Field(None, description="ID do usuário")
    previous_queries: List[Any] = Field(default_factory=list, description="Consultas anteriores (dicts)")
    previous_insights: List[Any] = Field(default_factory=list, description="Insights anteriores (dicts)")
    session_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados da sessão")

    created_at: datetime = Field(default_factory=datetime.now, description="Data de criação")
    last_accessed: datetime = Field(default_factory=datetime.now, description="Último acesso")

    class Config:
        arbitrary_types_allowed = True


class FinalResponse(BaseModel):
    """Resposta final completa para o usuário."""

    executive_summary: ExecutiveSummary = Field(description="Resumo executivo")
    consolidated_results: ConsolidatedResults = Field(description="Resultados consolidados")
    memory_context: Optional[MemoryContext] = Field(None, description="Contexto de memória")

    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados de processamento")
    total_processing_time: float = Field(description="Tempo total de processamento")

    success: bool = Field(True, description="Se a análise foi bem-sucedida")
    errors: List[str] = Field(default_factory=list, description="Erros encontrados")
    warnings: List[str] = Field(default_factory=list, description="Avisos gerados")