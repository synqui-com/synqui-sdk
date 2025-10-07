"""Framework-specific processors for hierarchical trace collection."""

from .base_processor import FrameworkProcessor, HierarchicalTrace
from .langchain_processor import LangChainProcessor

__all__ = ['FrameworkProcessor', 'HierarchicalTrace', 'LangChainProcessor']
