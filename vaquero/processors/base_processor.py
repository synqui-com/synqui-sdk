"""Base processor interface for framework-specific trace processing."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class HierarchicalTrace:
    """Canonical hierarchical trace format."""
    trace_id: str
    name: str
    agents: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]

class FrameworkProcessor(ABC):
    """Base interface for framework-specific processors."""
    
    @abstractmethod
    def add_span(self, span_data: Dict[str, Any]) -> None:
        """Add a span to the processor."""
        pass
    
    @abstractmethod
    def process_trace(self, trace_id: str) -> HierarchicalTrace:
        """Process all spans into hierarchical format."""
        pass
    
    @abstractmethod
    def detect_framework(self, span_data: Dict[str, Any]) -> bool:
        """Detect if this processor handles the given span."""
        pass
