"""
Modulo de agentes especializados para analise exploratoria de dados (EDA).

Este modulo contem todos os agentes especializados do sistema EDA:
- DataAnalyzerAgent: Analise descritiva e estatistica basica
- PatternDetectorAgent: Deteccao de padroes e tendencias
- AnomalyDetectorAgent: Identificacao de outliers e anomalias
- RelationshipAnalyzerAgent: Analise de correlacoes e relacionamentos
- ConclusionGeneratorAgent: Sintese de insights e conclusoes finais
- CodeGeneratorAgent: Geracao de codigo Python para analises EDA
"""

from .data_analyzer import DataAnalyzerAgent, get_data_analyzer_agent
from .pattern_detector import PatternDetectorAgent, get_pattern_detector_agent
from .anomaly_detector import AnomalyDetectorAgent, get_anomaly_detector_agent
from .relationship_analyzer import RelationshipAnalyzerAgent, get_relationship_analyzer_agent
from .conclusion_generator import ConclusionGeneratorAgent, get_conclusion_generator_agent
from .code_generator import CodeGeneratorAgent, get_code_generator_agent

__all__ = [
    # Classes dos agentes
    "DataAnalyzerAgent",
    "PatternDetectorAgent",
    "AnomalyDetectorAgent",
    "RelationshipAnalyzerAgent",
    "ConclusionGeneratorAgent",
    "CodeGeneratorAgent",

    # Funcoes singleton
    "get_data_analyzer_agent",
    "get_pattern_detector_agent",
    "get_anomaly_detector_agent",
    "get_relationship_analyzer_agent",
    "get_conclusion_generator_agent",
    "get_code_generator_agent"
]