"""
Schema para representação estruturada de consultas dos usuários.
Define modelos de dados para capturar intenção, entidades e contexto
das perguntas em linguagem natural sobre análise de dados.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .enums import EDAAnalysisType, QueryIntentType, VisualizationType


class QueryEntity(BaseModel):
    """Entidade extraída da consulta do usuário."""

    name: str = Field(description="Nome da entidade (ex: nome da coluna)")
    entity_type: str = Field(description="Tipo da entidade (column, value, operation)")
    value: Optional[str] = Field(None, description="Valor específico se aplicável")


class UserQuery(BaseModel):
    """Modelo para consulta do usuário."""

    query_text: str = Field(description="Texto original da consulta")
    intent_type: Optional[QueryIntentType] = Field(None, description="Tipo de intenção detectada")
    analysis_types: List[EDAAnalysisType] = Field(default_factory=list, description="Tipos de análise solicitados")
    entities: List[QueryEntity] = Field(default_factory=list, description="Entidades extraídas")
    requires_visualization: bool = Field(False, description="Se a consulta requer visualização")
    visualization_type: Optional[VisualizationType] = Field(None, description="Tipo de visualização sugerido")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais da consulta")


class QueryClassification(BaseModel):
    """Resultado da classificação de uma consulta."""

    query: UserQuery = Field(description="Consulta original classificada")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confiança da classificação")
    required_agents: List[EDAAnalysisType] = Field(description="Agentes necessários para responder")
    processing_order: List[EDAAnalysisType] = Field(description="Ordem de processamento recomendada")
    estimated_complexity: str = Field(description="Complexidade estimada (low, medium, high)")
    context_dependent: bool = Field(False, description="Se depende do contexto anterior")