"""
Gerenciador de memória e persistência de sessões.
Armazena contexto de conversas, histórico de análises, insights gerados
e estado de sessões para manter continuidade entre interações do usuário,
com suporte a diferentes backends de armazenamento.
"""

import json
import pickle
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import asdict
import hashlib

from .config import get_config
from .logger import get_logger, log_data_operation, log_error_with_context
from ..models.graph_schema import MemoryContext, DataInsight, ConsolidatedResults
from ..models.query_schema import UserQuery, QueryClassification


class MemoryError(Exception):
    """Erro específico para operações de memória."""
    pass


class MemoryManager:
    """Gerenciador centralizado de memória e persistência."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("memory_manager")

        # Configurar diretórios de armazenamento
        self.storage_path = Path("memory_storage")
        self.sessions_path = self.storage_path / "sessions"
        self.insights_path = self.storage_path / "insights"
        self.cache_path = self.storage_path / "cache"

        self._ensure_directories()

        # Cache em memória para sessions ativas
        self.active_sessions: Dict[str, MemoryContext] = {}

        # Configurações
        self.max_session_age_hours = 24
        self.max_insights_per_session = 100
        self.max_active_sessions = 50

    def _ensure_directories(self) -> None:
        """Cria diretórios de armazenamento se não existirem."""
        for path in [self.storage_path, self.sessions_path, self.insights_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self, user_id: Optional[str] = None) -> str:
        """Gera ID único para sessão."""
        timestamp = datetime.now().isoformat()
        base_string = f"{user_id or 'anonymous'}_{timestamp}"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]

    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Cria nova sessão de memória.

        Args:
            user_id: ID do usuário (opcional)

        Returns:
            ID da sessão criada
        """
        try:
            session_id = self._generate_session_id(user_id)

            memory_context = MemoryContext(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                previous_queries=[],
                previous_insights=[],
                session_metadata={"created": datetime.now().isoformat()}
            )

            # Armazenar em cache
            self.active_sessions[session_id] = memory_context

            # Persistir em disco IMEDIATAMENTE
            self._save_session(memory_context)

            # Verificar se foi salvo corretamente
            session_file = self.sessions_path / f"{session_id}.json"
            if not session_file.exists():
                self.logger.error(f"Falha ao persistir sessão {session_id} em disco")

            log_data_operation(
                "session_created",
                {"session_id": session_id, "user_id": user_id}
            )

            return session_id

        except Exception as e:
            log_error_with_context(e, {"operation": "create_session", "user_id": user_id})
            raise MemoryError(f"Erro ao criar sessão: {e}")

    def get_session(self, session_id: str) -> Optional[MemoryContext]:
        """
        Recupera contexto de sessão.

        Args:
            session_id: ID da sessão

        Returns:
            MemoryContext ou None se não encontrado
        """
        try:
            # Verificar cache primeiro
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_accessed = datetime.now()
                return session

            # Carregar do disco
            session_file = self.sessions_path / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Converter strings de datetime para objetos datetime
                if 'created_at' in data and isinstance(data['created_at'], str):
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                if 'last_accessed' in data and isinstance(data['last_accessed'], str):
                    data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])

                memory_context = MemoryContext(**data)
                memory_context.last_accessed = datetime.now()

                # Adicionar ao cache se há espaço
                if len(self.active_sessions) < self.max_active_sessions:
                    self.active_sessions[session_id] = memory_context

                return memory_context

            return None

        except Exception as e:
            log_error_with_context(e, {"operation": "get_session", "session_id": session_id})
            return None

    def save_query(self, session_id: str, query: UserQuery, classification: QueryClassification) -> None:
        """
        Salva consulta na sessão.

        Args:
            session_id: ID da sessão
            query: Consulta do usuário
            classification: Classificação da consulta
        """
        try:
            session = self.get_session(session_id)
            if not session:
                raise MemoryError(f"Sessão {session_id} não encontrada")

            # Adicionar consulta ao histórico
            query_entry = {
                "timestamp": datetime.now().isoformat(),
                "query_text": query.query_text,
                "intent_type": query.intent_type.value if query.intent_type else None,
                "analysis_types": [t.value for t in query.analysis_types],
                "requires_visualization": query.requires_visualization,
                "confidence_score": classification.confidence_score,
                "estimated_complexity": classification.estimated_complexity
            }

            session.previous_queries.append(query_entry)

            # Limitar histórico
            if len(session.previous_queries) > 50:
                session.previous_queries = session.previous_queries[-50:]

            # Atualizar metadados
            session.session_metadata["last_query"] = datetime.now().isoformat()
            session.last_accessed = datetime.now()

            # Salvar
            self._save_session(session)

            log_data_operation(
                "query_saved",
                {"session_id": session_id, "query_length": len(query.query_text)}
            )

        except Exception as e:
            log_error_with_context(e, {"operation": "save_query", "session_id": session_id})
            raise MemoryError(f"Erro ao salvar consulta: {e}")

    def save_insights(self, session_id: str, insights: List[DataInsight]) -> None:
        """
        Salva insights na sessão.

        Args:
            session_id: ID da sessão
            insights: Lista de insights
        """
        try:
            session = self.get_session(session_id)
            if not session:
                raise MemoryError(f"Sessão {session_id} não encontrada")

            # Converter insights para dict
            insight_dicts = []
            for insight in insights:
                insight_dict = {
                    "timestamp": datetime.now().isoformat(),
                    "insight_text": insight.insight_text,
                    "confidence": insight.confidence,
                    "analysis_type": insight.analysis_type.value,
                    "importance": insight.importance,
                    "related_columns": getattr(insight, 'related_columns', []),
                    "metadata": getattr(insight, 'metadata', {})
                }
                insight_dicts.append(insight_dict)

            # Adicionar ao histórico
            session.previous_insights.extend(insight_dicts)

            # Limitar insights por sessão
            if len(session.previous_insights) > self.max_insights_per_session:
                session.previous_insights = session.previous_insights[-self.max_insights_per_session:]

            # Atualizar metadados
            session.session_metadata["last_insights"] = datetime.now().isoformat()
            session.session_metadata["total_insights"] = len(session.previous_insights)
            session.last_accessed = datetime.now()

            # Salvar
            self._save_session(session)

            log_data_operation(
                "insights_saved",
                {"session_id": session_id, "insights_count": len(insights)}
            )

        except Exception as e:
            log_error_with_context(e, {"operation": "save_insights", "session_id": session_id})
            raise MemoryError(f"Erro ao salvar insights: {e}")

    def get_recent_insights(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recupera insights recentes da sessão.

        Args:
            session_id: ID da sessão
            limit: Número máximo de insights

        Returns:
            Lista de insights recentes
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return []

            # Retornar insights mais recentes
            recent_insights = session.previous_insights[-limit:] if session.previous_insights else []
            return list(reversed(recent_insights))  # Mais recentes primeiro

        except Exception as e:
            log_error_with_context(e, {"operation": "get_recent_insights", "session_id": session_id})
            return []

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Recupera contexto resumido da sessão para LLM.

        Args:
            session_id: ID da sessão

        Returns:
            Dict com contexto da sessão
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return {}

            # Últimas 5 consultas
            recent_queries = session.previous_queries[-5:] if session.previous_queries else []

            # Últimos 10 insights mais relevantes
            recent_insights = self.get_recent_insights(session_id, 10)
            high_conf_insights = [
                i for i in recent_insights
                if i.get('confidence', 0) > 0.7
            ][:5]

            context = {
                "session_id": session_id,
                "session_age_hours": (datetime.now() - session.created_at).total_seconds() / 3600,
                "total_queries": len(session.previous_queries),
                "total_insights": len(session.previous_insights),
                "recent_queries": [q.get('query_text', '') for q in recent_queries],
                "recent_query_types": [q.get('intent_type') for q in recent_queries if q.get('intent_type')],
                "high_confidence_insights": [i.get('insight_text', '') for i in high_conf_insights],
                "analysis_patterns": self._analyze_session_patterns(session)
            }

            return context

        except Exception as e:
            log_error_with_context(e, {"operation": "get_session_context", "session_id": session_id})
            return {}

    def _analyze_session_patterns(self, session: MemoryContext) -> Dict[str, Any]:
        """Analisa padrões na sessão."""
        patterns = {
            "common_analysis_types": [],
            "frequent_topics": [],
            "complexity_trend": "stable"
        }

        try:
            if not session.previous_queries:
                return patterns

            # Análise de tipos de análise mais comuns
            analysis_types = []
            for query in session.previous_queries:
                analysis_types.extend(query.get('analysis_types', []))

            if analysis_types:
                from collections import Counter
                common_types = Counter(analysis_types).most_common(3)
                patterns["common_analysis_types"] = [t[0] for t in common_types]

            # Tendência de complexidade
            complexities = [q.get('estimated_complexity') for q in session.previous_queries[-5:]]
            complexities = [c for c in complexities if c]

            if len(complexities) >= 2:
                if complexities[-1] == 'high' and complexities[0] != 'high':
                    patterns["complexity_trend"] = "increasing"
                elif complexities[-1] == 'low' and complexities[0] != 'low':
                    patterns["complexity_trend"] = "decreasing"

        except Exception as e:
            self.logger.warning(f"Erro ao analisar padrões da sessão: {e}")

        return patterns

    def _save_session(self, session: MemoryContext) -> None:
        """Salva sessão em disco."""
        try:
            session_file = self.sessions_path / f"{session.session_id}.json"

            # Converter para dict serializável
            session_dict = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "previous_queries": session.previous_queries,
                "previous_insights": session.previous_insights,
                "session_metadata": session.session_metadata
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False)

        except Exception as e:
            log_error_with_context(e, {"operation": "save_session", "session_id": session.session_id})

    def cleanup_old_sessions(self) -> int:
        """
        Remove sessões antigas.

        Returns:
            Número de sessões removidas
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.max_session_age_hours)
            removed_count = 0

            # Limpar cache
            to_remove = []
            for session_id, session in self.active_sessions.items():
                if session.last_accessed < cutoff_time:
                    to_remove.append(session_id)

            for session_id in to_remove:
                del self.active_sessions[session_id]
                removed_count += 1

            # Limpar arquivos
            for session_file in self.sessions_path.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    last_accessed = datetime.fromisoformat(data.get('last_accessed', ''))
                    if last_accessed < cutoff_time:
                        session_file.unlink()
                        removed_count += 1

                except Exception as e:
                    self.logger.warning(f"Erro ao verificar arquivo {session_file}: {e}")

            log_data_operation(
                "sessions_cleaned",
                {"removed_count": removed_count, "cutoff_hours": self.max_session_age_hours}
            )

            return removed_count

        except Exception as e:
            log_error_with_context(e, {"operation": "cleanup_old_sessions"})
            return 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das sessões."""
        try:
            active_count = len(self.active_sessions)
            total_files = len(list(self.sessions_path.glob("*.json")))

            # Calcular idade média das sessões ativas
            if self.active_sessions:
                ages = [
                    (datetime.now() - session.created_at).total_seconds() / 3600
                    for session in self.active_sessions.values()
                ]
                avg_age_hours = sum(ages) / len(ages)
            else:
                avg_age_hours = 0

            return {
                "active_sessions": active_count,
                "total_session_files": total_files,
                "average_session_age_hours": round(avg_age_hours, 2),
                "storage_path": str(self.storage_path),
                "max_session_age_hours": self.max_session_age_hours
            }

        except Exception as e:
            log_error_with_context(e, {"operation": "get_session_stats"})
            return {"error": str(e)}

    def search_insights(self, query: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Busca insights por texto.

        Args:
            query: Texto para buscar
            session_id: ID da sessão (opcional, busca em todas se None)

        Returns:
            Lista de insights encontrados
        """
        try:
            results = []
            query_lower = query.lower()

            if session_id:
                # Buscar apenas na sessão específica
                session = self.get_session(session_id)
                if session:
                    for insight in session.previous_insights:
                        if query_lower in insight.get('insight_text', '').lower():
                            results.append(insight)
            else:
                # Buscar em todas as sessões ativas
                for session in self.active_sessions.values():
                    for insight in session.previous_insights:
                        if query_lower in insight.get('insight_text', '').lower():
                            insight_copy = insight.copy()
                            insight_copy['session_id'] = session.session_id
                            results.append(insight_copy)

            # Ordenar por confiança
            results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return results[:20]  # Limitar a 20 resultados

        except Exception as e:
            log_error_with_context(e, {"operation": "search_insights", "query": query})
            return []


# Instância singleton
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Obtém instância singleton do MemoryManager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager