"""
Módulo interface - Componentes de apoio para a interface Streamlit.

Este módulo contém componentes reutilizáveis, gerenciadores de sessão,
renderizadores de visualização e handlers de erro específicos para
a interface web do EDA Agent.
"""

try:
    from .streamlit_components import (
        render_metric_card,
        render_analysis_summary,
        render_data_preview
    )
except ImportError:
    # Fallback se houver problemas de importação
    render_metric_card = None
    render_analysis_summary = None
    render_data_preview = None
try:
    from .session_manager import StreamlitSessionManager
except ImportError:
    StreamlitSessionManager = None

try:
    from .visualization_renderer import VisualizationRenderer
except ImportError:
    VisualizationRenderer = None

try:
    from .error_handler import StreamlitErrorHandler
except ImportError:
    StreamlitErrorHandler = None

__all__ = [
    "render_metric_card",
    "render_analysis_summary",
    "render_data_preview",
    "StreamlitSessionManager",
    "VisualizationRenderer",
    "StreamlitErrorHandler"
]