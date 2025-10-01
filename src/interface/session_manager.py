"""
Gerenciador de sessão para Streamlit.

Fornece funcionalidades avançadas de gerenciamento de estado,
cache e persistência de dados na sessão do usuário.
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import hashlib


class StreamlitSessionManager:
    """
    Gerenciador avançado de sessão para o EDA Agent.

    Fornece funcionalidades de cache, persistência e
    gerenciamento de estado específicos para a aplicação.
    """

    def __init__(self):
        self.session_key_prefix = "eda_agent_"
        self._initialize_session()

    def _initialize_session(self):
        """Inicializa estruturas básicas da sessão."""
        # Metadados da sessão
        if f"{self.session_key_prefix}metadata" not in st.session_state:
            st.session_state[f"{self.session_key_prefix}metadata"] = {
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "session_id": self._generate_session_id(),
                "version": "1.0.0"
            }

        # Cache de resultados
        if f"{self.session_key_prefix}cache" not in st.session_state:
            st.session_state[f"{self.session_key_prefix}cache"] = {}

        # Histórico de atividades
        if f"{self.session_key_prefix}activity_log" not in st.session_state:
            st.session_state[f"{self.session_key_prefix}activity_log"] = []

    def _generate_session_id(self) -> str:
        """Gera um ID único para a sessão."""
        import uuid
        return str(uuid.uuid4())[:12]

    def get_session_metadata(self) -> Dict[str, Any]:
        """Retorna metadados da sessão."""
        return st.session_state.get(f"{self.session_key_prefix}metadata", {})

    def update_last_activity(self):
        """Atualiza timestamp da última atividade."""
        metadata = self.get_session_metadata()
        metadata["last_activity"] = datetime.now().isoformat()
        st.session_state[f"{self.session_key_prefix}metadata"] = metadata

    def get_session_duration(self) -> timedelta:
        """Calcula duração da sessão atual."""
        metadata = self.get_session_metadata()
        created_at = datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat()))
        return datetime.now() - created_at

    def log_activity(self, activity_type: str, details: Dict[str, Any]):
        """
        Registra uma atividade no log da sessão.

        Args:
            activity_type: Tipo de atividade
            details: Detalhes da atividade
        """
        activity_log = st.session_state.get(f"{self.session_key_prefix}activity_log", [])

        activity_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "details": details
        }

        activity_log.append(activity_entry)

        # Manter apenas últimas 100 atividades
        if len(activity_log) > 100:
            activity_log = activity_log[-100:]

        st.session_state[f"{self.session_key_prefix}activity_log"] = activity_log
        self.update_last_activity()

    def get_activity_log(self, activity_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retorna log de atividades.

        Args:
            activity_type: Filtrar por tipo (opcional)
            limit: Limite de registros

        Returns:
            Lista de atividades
        """
        activity_log = st.session_state.get(f"{self.session_key_prefix}activity_log", [])

        if activity_type:
            activity_log = [a for a in activity_log if a.get("type") == activity_type]

        return activity_log[-limit:]

    def cache_result(self, key: str, data: Any, ttl_minutes: int = 60):
        """
        Armazena resultado no cache da sessão.

        Args:
            key: Chave única para o cache
            data: Dados a armazenar
            ttl_minutes: Tempo de vida em minutos
        """
        cache = st.session_state.get(f"{self.session_key_prefix}cache", {})

        cache_entry = {
            "data": data,
            "created_at": datetime.now().isoformat(),
            "ttl_minutes": ttl_minutes
        }

        cache[key] = cache_entry
        st.session_state[f"{self.session_key_prefix}cache"] = cache

        self.log_activity("cache_write", {"key": key, "ttl": ttl_minutes})

    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Recupera resultado do cache se ainda válido.

        Args:
            key: Chave do cache

        Returns:
            Dados armazenados ou None se expirado/inexistente
        """
        cache = st.session_state.get(f"{self.session_key_prefix}cache", {})
        cache_entry = cache.get(key)

        if not cache_entry:
            return None

        # Verificar se ainda é válido
        created_at = datetime.fromisoformat(cache_entry["created_at"])
        ttl = timedelta(minutes=cache_entry["ttl_minutes"])

        if datetime.now() > created_at + ttl:
            # Cache expirado, remover
            del cache[key]
            st.session_state[f"{self.session_key_prefix}cache"] = cache
            self.log_activity("cache_expired", {"key": key})
            return None

        self.log_activity("cache_hit", {"key": key})
        return cache_entry["data"]

    def clear_cache(self, pattern: Optional[str] = None):
        """
        Limpa cache da sessão.

        Args:
            pattern: Padrão para filtrar chaves (opcional)
        """
        cache = st.session_state.get(f"{self.session_key_prefix}cache", {})

        if pattern:
            # Remover apenas chaves que correspondem ao padrão
            keys_to_remove = [k for k in cache.keys() if pattern in k]
            for key in keys_to_remove:
                del cache[key]
        else:
            # Limpar todo o cache
            cache.clear()

        st.session_state[f"{self.session_key_prefix}cache"] = cache
        self.log_activity("cache_clear", {"pattern": pattern})

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        cache = st.session_state.get(f"{self.session_key_prefix}cache", {})

        total_entries = len(cache)
        valid_entries = 0
        expired_entries = 0

        for entry in cache.values():
            created_at = datetime.fromisoformat(entry["created_at"])
            ttl = timedelta(minutes=entry["ttl_minutes"])

            if datetime.now() <= created_at + ttl:
                valid_entries += 1
            else:
                expired_entries += 1

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "hit_rate": self._calculate_hit_rate()
        }

    def _calculate_hit_rate(self) -> float:
        """Calcula taxa de acerto do cache."""
        activities = self.get_activity_log()
        cache_activities = [a for a in activities if a["type"] in ["cache_hit", "cache_miss"]]

        if not cache_activities:
            return 0.0

        hits = len([a for a in cache_activities if a["type"] == "cache_hit"])
        return hits / len(cache_activities) * 100

    def set_user_preference(self, key: str, value: Any):
        """
        Define preferência do usuário.

        Args:
            key: Chave da preferência
            value: Valor da preferência
        """
        prefs_key = f"{self.session_key_prefix}preferences"
        preferences = st.session_state.get(prefs_key, {})
        preferences[key] = value
        st.session_state[prefs_key] = preferences

        self.log_activity("preference_set", {"key": key})

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """
        Obtém preferência do usuário.

        Args:
            key: Chave da preferência
            default: Valor padrão

        Returns:
            Valor da preferência ou padrão
        """
        prefs_key = f"{self.session_key_prefix}preferences"
        preferences = st.session_state.get(prefs_key, {})
        return preferences.get(key, default)

    def export_session_data(self) -> Dict[str, Any]:
        """
        Exporta dados da sessão para backup/análise.

        Returns:
            Dicionário com dados da sessão
        """
        session_data = {}

        for key, value in st.session_state.items():
            if key.startswith(self.session_key_prefix):
                try:
                    # Tentar serializar para JSON para verificar compatibilidade
                    json.dumps(value, default=str)
                    session_data[key] = value
                except (TypeError, ValueError):
                    # Pular dados não serializáveis
                    session_data[key] = f"<não-serializável: {type(value).__name__}>"

        return session_data

    def cleanup_expired_cache(self):
        """Remove entradas expiradas do cache."""
        cache = st.session_state.get(f"{self.session_key_prefix}cache", {})
        keys_to_remove = []

        for key, entry in cache.items():
            created_at = datetime.fromisoformat(entry["created_at"])
            ttl = timedelta(minutes=entry["ttl_minutes"])

            if datetime.now() > created_at + ttl:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del cache[key]

        if keys_to_remove:
            st.session_state[f"{self.session_key_prefix}cache"] = cache
            self.log_activity("cache_cleanup", {"removed_count": len(keys_to_remove)})

    def reset_session(self, keep_preferences: bool = True):
        """
        Reseta a sessão mantendo opcionalmente as preferências.

        Args:
            keep_preferences: Se deve manter preferências do usuário
        """
        preferences = None
        if keep_preferences:
            prefs_key = f"{self.session_key_prefix}preferences"
            preferences = st.session_state.get(prefs_key, {})

        # Remover todas as chaves da sessão EDA
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith(self.session_key_prefix)]
        for key in keys_to_remove:
            del st.session_state[key]

        # Reinicializar
        self._initialize_session()

        # Restaurar preferências se solicitado
        if keep_preferences and preferences:
            st.session_state[f"{self.session_key_prefix}preferences"] = preferences

        self.log_activity("session_reset", {"keep_preferences": keep_preferences})