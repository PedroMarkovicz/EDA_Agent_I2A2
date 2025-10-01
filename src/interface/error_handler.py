"""
Handler especializado de erros para interface Streamlit.

Fornece funcionalidades avançadas de captura, formatação
e exibição de erros na interface do usuário.
"""

import streamlit as st
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import logging


class ErrorSeverity(Enum):
    """Níveis de severidade de erro."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StreamlitErrorHandler:
    """
    Handler especializado para erros na interface Streamlit.

    Fornece funcionalidades de captura, logging, formatação
    e exibição inteligente de erros para o usuário.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history = []

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: Optional[str] = None,
        show_details: bool = True,
        suggestions: Optional[List[str]] = None
    ):
        """
        Manipula e exibe um erro de forma inteligente.

        Args:
            error: Exceção capturada
            context: Contexto do erro
            severity: Severidade do erro
            user_message: Mensagem personalizada para o usuário
            show_details: Se deve mostrar detalhes técnicos
            suggestions: Sugestões de resolução
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "severity": severity.value,
            "traceback": traceback.format_exc() if show_details else None
        }

        # Adicionar ao histórico
        self.error_history.append(error_info)

        # Log do erro
        self._log_error(error_info)

        # Exibir para o usuário
        self._display_error(error_info, user_message, show_details, suggestions)

    def _log_error(self, error_info: Dict[str, Any]):
        """Registra erro no sistema de logging."""
        severity = error_info["severity"]
        message = f"[{severity.upper()}] {error_info['type']}: {error_info['message']}"

        if severity == ErrorSeverity.CRITICAL.value:
            self.logger.critical(message, extra=error_info)
        elif severity == ErrorSeverity.HIGH.value:
            self.logger.error(message, extra=error_info)
        elif severity == ErrorSeverity.MEDIUM.value:
            self.logger.warning(message, extra=error_info)
        else:
            self.logger.info(message, extra=error_info)

    def _display_error(
        self,
        error_info: Dict[str, Any],
        user_message: Optional[str],
        show_details: bool,
        suggestions: Optional[List[str]]
    ):
        """Exibe erro na interface Streamlit."""
        severity = error_info["severity"]

        # Definir ícone e cor baseado na severidade
        if severity == ErrorSeverity.CRITICAL.value:
            icon = "🚨"
            st.error(f"{icon} Erro Crítico")
        elif severity == ErrorSeverity.HIGH.value:
            icon = "❌"
            st.error(f"{icon} Erro")
        elif severity == ErrorSeverity.MEDIUM.value:
            icon = "⚠️"
            st.warning(f"{icon} Aviso")
        else:
            icon = "ℹ️"
            st.info(f"{icon} Informação")

        # Mensagem principal
        if user_message:
            st.markdown(f"**{user_message}**")
        else:
            st.markdown(f"**{error_info['message']}**")

        # Sugestões de resolução
        if suggestions:
            st.markdown("**💡 Sugestões:**")
            for suggestion in suggestions:
                st.markdown(f"• {suggestion}")

        # Detalhes técnicos (expandível)
        if show_details:
            with st.expander("🔧 Detalhes técnicos", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Tipo:**")
                    st.code(error_info["type"])
                    st.markdown("**Horário:**")
                    st.text(error_info["timestamp"])

                with col2:
                    st.markdown("**Severidade:**")
                    st.text(error_info["severity"].upper())

                if error_info["context"]:
                    st.markdown("**Contexto:**")
                    st.json(error_info["context"])

                if error_info["traceback"]:
                    st.markdown("**Traceback:**")
                    st.code(error_info["traceback"], language="python")

    def handle_csv_upload_error(self, error: Exception, filename: str):
        """Handler específico para erros de upload de CSV."""
        suggestions = [
            "Verifique se o arquivo é um CSV válido",
            "Certifique-se de que o arquivo não está corrompido",
            "Tente usar encoding UTF-8 ou Latin-1",
            "Verifique se o arquivo não excede o limite de tamanho"
        ]

        self.handle_error(
            error=error,
            context={"operation": "csv_upload", "filename": filename},
            severity=ErrorSeverity.MEDIUM,
            user_message=f"Erro ao carregar arquivo '{filename}'",
            suggestions=suggestions
        )

    def handle_analysis_error(self, error: Exception, query: str, analysis_type: str):
        """Handler específico para erros de análise."""
        suggestions = [
            "Verifique se sua pergunta está clara e específica",
            "Certifique-se de que os dados são adequados para este tipo de análise",
            "Tente reformular sua pergunta",
            "Verifique se há dados suficientes para a análise"
        ]

        self.handle_error(
            error=error,
            context={
                "operation": "analysis",
                "query": query,
                "analysis_type": analysis_type
            },
            severity=ErrorSeverity.HIGH,
            user_message="Erro durante a análise dos dados",
            suggestions=suggestions
        )

    def handle_visualization_error(self, error: Exception, viz_type: str):
        """Handler específico para erros de visualização."""
        suggestions = [
            "Verifique se os dados são adequados para este tipo de visualização",
            "Tente um tipo diferente de gráfico",
            "Certifique-se de que não há dados ausentes nos campos necessários",
            "Recarregue a página se o problema persistir"
        ]

        self.handle_error(
            error=error,
            context={"operation": "visualization", "viz_type": viz_type},
            severity=ErrorSeverity.MEDIUM,
            user_message="Erro ao gerar visualização",
            suggestions=suggestions
        )

    def handle_workflow_error(self, error: Exception, workflow_step: str):
        """Handler específico para erros do workflow."""
        suggestions = [
            "Tente novamente em alguns momentos",
            "Verifique sua conexão com a internet",
            "Simplifique sua pergunta",
            "Recarregue a página se necessário"
        ]

        self.handle_error(
            error=error,
            context={"operation": "workflow", "step": workflow_step},
            severity=ErrorSeverity.HIGH,
            user_message="Erro no processamento da análise",
            suggestions=suggestions
        )

    def handle_system_error(self, error: Exception, component: str):
        """Handler específico para erros do sistema."""
        suggestions = [
            "Recarregue a página",
            "Limpe o cache do navegador",
            "Tente novamente mais tarde",
            "Entre em contato com o suporte se o problema persistir"
        ]

        self.handle_error(
            error=error,
            context={"operation": "system", "component": component},
            severity=ErrorSeverity.CRITICAL,
            user_message="Erro interno do sistema",
            suggestions=suggestions
        )

    def get_error_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Retorna resumo dos erros recentes.

        Args:
            time_window_minutes: Janela de tempo em minutos

        Returns:
            Resumo dos erros
        """
        cutoff_time = datetime.now().timestamp() - (time_window_minutes * 60)
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error["timestamp"]).timestamp() > cutoff_time
        ]

        summary = {
            "total_errors": len(recent_errors),
            "by_severity": {},
            "by_type": {},
            "most_recent": recent_errors[-1] if recent_errors else None
        }

        # Agrupar por severidade
        for error in recent_errors:
            severity = error["severity"]
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

        # Agrupar por tipo
        for error in recent_errors:
            error_type = error["type"]
            summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1

        return summary

    def display_error_dashboard(self):
        """Exibe dashboard de erros para debugging."""
        st.subheader("🐛 Dashboard de Erros")

        if not self.error_history:
            st.info("Nenhum erro registrado nesta sessão")
            return

        # Resumo geral
        summary = self.get_error_summary()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Erros", summary["total_errors"])
        with col2:
            critical_count = summary["by_severity"].get("critical", 0)
            st.metric("Erros Críticos", critical_count)
        with col3:
            if summary["most_recent"]:
                last_error_time = summary["most_recent"]["timestamp"]
                st.metric("Último Erro", last_error_time[-8:])  # HH:MM:SS

        # Lista de erros recentes
        with st.expander("Histórico de Erros", expanded=False):
            for i, error in enumerate(reversed(self.error_history[-10:])):  # Últimos 10
                st.markdown(f"**{i+1}. {error['type']}** ({error['severity']})")
                st.text(f"📅 {error['timestamp']}")
                st.text(f"💬 {error['message']}")
                if error['context']:
                    st.json(error['context'])
                st.divider()

    def clear_error_history(self):
        """Limpa histórico de erros."""
        self.error_history = []

    def export_error_log(self) -> str:
        """
        Exporta log de erros para análise.

        Returns:
            String JSON com histórico de erros
        """
        import json
        return json.dumps(self.error_history, indent=2, default=str)