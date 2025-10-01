"""
Modelos para resultados intermediários de análises de dados.
Define estruturas para encapsular dataframes processados, análises estatísticas,
gráficos gerados e outros outputs dos agentes especializados.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime

from .enums import EDAAnalysisType, ProcessingStatus, VisualizationType


class StatisticalSummary(BaseModel):
    """Resumo estatístico de uma variável."""

    column_name: str = Field(description="Nome da coluna")
    data_type: str = Field(description="Tipo de dados (numeric, categorical, datetime)")
    count: int = Field(description="Número de valores não nulos")
    missing_count: int = Field(description="Número de valores ausentes")
    missing_percentage: float = Field(description="Porcentagem de valores ausentes")
    unique_count: int = Field(description="Número de valores únicos")

    # Estatísticas para dados numéricos
    mean: Optional[float] = Field(None, description="Média")
    median: Optional[float] = Field(None, description="Mediana")
    std: Optional[float] = Field(None, description="Desvio padrão")
    min_value: Optional[Union[float, str]] = Field(None, description="Valor mínimo")
    max_value: Optional[Union[float, str]] = Field(None, description="Valor máximo")
    q25: Optional[float] = Field(None, description="Primeiro quartil")
    q75: Optional[float] = Field(None, description="Terceiro quartil")

    # Para dados categóricos
    mode: Optional[Union[str, float]] = Field(None, description="Valor mais frequente")
    mode_frequency: Optional[int] = Field(None, description="Frequência do valor mais comum")


class VisualizationResult(BaseModel):
    """Resultado de uma visualização gerada."""

    visualization_type: VisualizationType = Field(description="Tipo de visualização")
    title: str = Field(description="Título do gráfico")
    description: str = Field(description="Descrição da visualização")
    file_path: Optional[str] = Field(None, description="Caminho do arquivo gerado")
    base64_image: Optional[str] = Field(None, description="Imagem em base64")
    plotly_json: Optional[Dict] = Field(None, description="JSON do gráfico Plotly")
    columns_used: List[str] = Field(description="Colunas utilizadas na visualização")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados do gráfico")


class DescriptiveAnalysisResult(BaseModel):
    """Resultado da análise descritiva."""

    analysis_type: EDAAnalysisType = Field(EDAAnalysisType.DESCRIPTIVE)
    status: ProcessingStatus = Field(ProcessingStatus.PENDING)
    dataset_overview: Dict[str, Any] = Field(default_factory=dict, description="Visão geral do dataset")
    column_summaries: List[StatisticalSummary] = Field(default_factory=list, description="Resumos por coluna")
    data_types_summary: Dict[str, int] = Field(default_factory=dict, description="Contagem por tipo de dados")
    missing_data_summary: Dict[str, Any] = Field(default_factory=dict, description="Resumo de dados ausentes")
    visualizations: List[VisualizationResult] = Field(default_factory=list, description="Visualizações geradas")
    insights: List[str] = Field(default_factory=list, description="Insights descobertos")
    processing_time: Optional[float] = Field(None, description="Tempo de processamento em segundos")


class PatternAnalysisResult(BaseModel):
    """Resultado da análise de padrões."""

    analysis_type: EDAAnalysisType = Field(EDAAnalysisType.PATTERN)
    status: ProcessingStatus = Field(ProcessingStatus.PENDING)
    temporal_patterns: Dict[str, Any] = Field(default_factory=dict, description="Padrões temporais detectados")
    frequency_analysis: Dict[str, Any] = Field(default_factory=dict, description="Análise de frequências")
    clustering_results: Dict[str, Any] = Field(default_factory=dict, description="Resultados de clustering")
    trends: List[str] = Field(default_factory=list, description="Tendências identificadas")
    visualizations: List[VisualizationResult] = Field(default_factory=list, description="Visualizações geradas")
    insights: List[str] = Field(default_factory=list, description="Insights descobertos")
    processing_time: Optional[float] = Field(None, description="Tempo de processamento em segundos")


class AnomalyAnalysisResult(BaseModel):
    """Resultado da análise de anomalias."""

    analysis_type: EDAAnalysisType = Field(EDAAnalysisType.ANOMALY)
    status: ProcessingStatus = Field(ProcessingStatus.PENDING)
    outlier_detection_methods: List[str] = Field(default_factory=list, description="Métodos utilizados")
    outliers_by_column: Dict[str, List[Any]] = Field(default_factory=dict, description="Outliers por coluna")
    outlier_statistics: Dict[str, Any] = Field(default_factory=dict, description="Estatísticas dos outliers")
    impact_analysis: Dict[str, Any] = Field(default_factory=dict, description="Análise do impacto")
    recommendations: List[str] = Field(default_factory=list, description="Recomendações de tratamento")
    visualizations: List[VisualizationResult] = Field(default_factory=list, description="Visualizações geradas")
    insights: List[str] = Field(default_factory=list, description="Insights descobertos")
    processing_time: Optional[float] = Field(None, description="Tempo de processamento em segundos")


class RelationshipAnalysisResult(BaseModel):
    """Resultado da análise de relacionamentos."""

    analysis_type: EDAAnalysisType = Field(EDAAnalysisType.RELATIONSHIP)
    status: ProcessingStatus = Field(ProcessingStatus.PENDING)
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Matriz de correlação")
    strong_correlations: List[Dict[str, Any]] = Field(default_factory=list, description="Correlações fortes")
    weak_correlations: List[Dict[str, Any]] = Field(default_factory=list, description="Correlações fracas")
    variable_dependencies: Dict[str, Any] = Field(default_factory=dict, description="Dependências entre variáveis")
    multicollinearity_analysis: Dict[str, Any] = Field(default_factory=dict, description="Análise de multicolinearidade")
    visualizations: List[VisualizationResult] = Field(default_factory=list, description="Visualizações geradas")
    insights: List[str] = Field(default_factory=list, description="Insights descobertos")
    processing_time: Optional[float] = Field(None, description="Tempo de processamento em segundos")


class AnalysisConclusion(BaseModel):
    """Conclusão final da análise EDA."""

    summary: str = Field(description="Resumo executivo da análise")
    key_findings: List[str] = Field(default_factory=list, description="Principais descobertas")
    recommendations: List[str] = Field(default_factory=list, description="Recomendações de ação")
    data_quality_assessment: Dict[str, Any] = Field(default_factory=dict, description="Avaliação da qualidade dos dados")
    business_impact: str = Field(description="Impacto potencial no negócio")
    next_steps: List[str] = Field(default_factory=list, description="Próximos passos sugeridos")
    confidence_score: float = Field(ge=0, le=1, description="Confiança na análise (0-1)")
    analysis_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados da conclusão")