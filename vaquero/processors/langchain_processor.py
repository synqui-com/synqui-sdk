"""LangChain-specific processor for hierarchical trace collection."""

import logging
from typing import Dict, List, Any, Optional
from .base_processor import FrameworkProcessor, HierarchicalTrace

logger = logging.getLogger(__name__)

class LangChainProcessor(FrameworkProcessor):
    """LangChain processor that groups internal components under logical agents."""
    
    def __init__(self):
        self.logical_agents = {}  # logical_agent_name -> agent_data
        self.internal_components = {}  # logical_agent_name -> [components]
        self.spans = []  # All spans for this trace
    
    def add_span(self, span_data: Dict[str, Any]) -> None:
        """Add LangChain span to processor."""
        self.spans.append(span_data)
        
        agent_name = span_data.get('agent_name', '')
        logical_agent = self._determine_logical_agent(span_data)
        
        if logical_agent not in self.logical_agents:
            self.logical_agents[logical_agent] = {
                'name': logical_agent,
                'level': 1,
                'framework': 'langchain',
                'component_type': 'agent',
                'parent_agent_id': None,
                'spans': []
            }
            self.internal_components[logical_agent] = []
        
        # Add to logical agent
        self.logical_agents[logical_agent]['spans'].append(span_data)
        self.internal_components[logical_agent].append(span_data)
        
        logger.debug(f"Added {agent_name} to logical agent {logical_agent}")
    
    def _determine_logical_agent(self, span_data: Dict[str, Any]) -> str:
        """Determine which logical agent this span belongs to."""
        agent_name = span_data.get('agent_name', '')
        metadata = span_data.get('metadata', {})
        
        # Check for explicit stage in metadata
        stage = metadata.get('stage')
        if stage:
            if stage == 'validation':
                return 'validation_agent'
            elif stage == 'analysis':
                return 'analysis_agent'
            elif stage == 'report_generation':
                return 'report_agent'
        
        # Check agent name patterns
        if 'validation' in agent_name.lower():
            return 'validation_agent'
        elif 'analysis' in agent_name.lower():
            return 'analysis_agent'
        elif 'report' in agent_name.lower():
            return 'report_agent'
        elif 'workflow' in agent_name.lower():
            return 'workflow_root'
        
        # For internal LangChain components, try to determine parent from context
        # This is the key intelligence - we need to group internal components
        # under their logical agents based on execution context
        return self._determine_parent_from_context(span_data)
    
    def _determine_parent_from_context(self, span_data: Dict[str, Any]) -> str:
        """Determine parent logical agent from execution context."""
        # This is where we implement the intelligence to group internal components
        # For now, we'll use a simple heuristic based on span order and metadata
        
        # If this is a tool call, it likely belongs to the most recent logical agent
        agent_name = span_data.get('agent_name', '')
        
        if agent_name.startswith('tool:'):
            # Tool calls should belong to the logical agent that called them
            # We'll use the most recent logical agent as a heuristic
            if 'validation_agent' in self.logical_agents:
                return 'validation_agent'
            elif 'analysis_agent' in self.logical_agents:
                return 'analysis_agent'
            elif 'report_agent' in self.logical_agents:
                return 'report_agent'
        
        # For other internal components, use the same logic
        if 'validation_agent' in self.logical_agents:
            return 'validation_agent'
        elif 'analysis_agent' in self.logical_agents:
            return 'analysis_agent'
        elif 'report_agent' in self.logical_agents:
            return 'report_agent'
        
        # Default to unknown for now
        return 'unknown_agent'
    
    def process_trace(self, trace_id: str) -> HierarchicalTrace:
        """Process all spans into hierarchical format."""
        agents = []
        dependencies = []
        
        # Create logical agents
        for logical_agent_name, agent_data in self.logical_agents.items():
            # Create logical agent
            logical_agent = {
                'trace_id': trace_id,
                'agent_id': f"{logical_agent_name}_{trace_id}",
                'name': logical_agent_name,
                'level': 1,
                'framework': 'langchain',
                'component_type': 'agent',
                'parent_agent_id': None,
                'status': 'completed',
                'start_time': self._get_earliest_start_time(agent_data['spans']),
                'end_time': self._get_latest_end_time(agent_data['spans']),
                'duration_ms': self._calculate_duration(agent_data['spans']),
                'input_tokens': sum(s.get('input_tokens', 0) for s in agent_data['spans']),
                'output_tokens': sum(s.get('output_tokens', 0) for s in agent_data['spans']),
                'total_tokens': sum(s.get('total_tokens', 0) for s in agent_data['spans']),
                'cost': sum(s.get('cost', 0.0) for s in agent_data['spans']),
                'tags': {},
                'input_data': {},
                'output_data': {},
                'metadata': {},
                'framework_metadata': {}
            }
            agents.append(logical_agent)
            
            # Create internal components
            for span in agent_data['spans']:
                component = {
                    'trace_id': trace_id,
                    'agent_id': span.get('span_id'),
                    'name': span.get('agent_name', ''),
                    'level': 2,
                    'framework': 'langchain',
                    'component_type': self._get_component_type(span),
                    'parent_agent_id': logical_agent['agent_id'],
                    'status': span.get('status', 'completed'),
                    'start_time': span.get('start_time'),
                    'end_time': span.get('end_time'),
                    'duration_ms': span.get('duration_ms', 0),
                    'input_tokens': span.get('input_tokens', 0),
                    'output_tokens': span.get('output_tokens', 0),
                    'total_tokens': span.get('total_tokens', 0),
                    'cost': span.get('cost', 0.0),
                    'tags': span.get('tags', {}),
                    'input_data': span.get('inputs', {}),
                    'output_data': span.get('outputs', {}),
                    'metadata': span.get('metadata', {}),
                    'framework_metadata': span.get('tags', {})
                }
                agents.append(component)
        
        return HierarchicalTrace(
            trace_id=trace_id,
            name='langchain_workflow',
            agents=agents,
            dependencies=dependencies
        )
    
    def _get_component_type(self, span: Dict[str, Any]) -> str:
        """Determine component type from span data."""
        agent_name = span.get('agent_name', '')
        if agent_name.startswith('langchain:'):
            return 'chain'
        elif agent_name.startswith('llm:'):
            return 'llm'
        elif agent_name.startswith('tool:'):
            return 'tool'
        else:
            return 'component'
    
    def _get_earliest_start_time(self, spans: List[Dict[str, Any]]) -> str:
        """Get earliest start time from spans."""
        start_times = [s.get('start_time') for s in spans if s.get('start_time')]
        return min(start_times) if start_times else None
    
    def _get_latest_end_time(self, spans: List[Dict[str, Any]]) -> str:
        """Get latest end time from spans."""
        end_times = [s.get('end_time') for s in spans if s.get('end_time')]
        return max(end_times) if end_times else None
    
    def _calculate_duration(self, spans: List[Dict[str, Any]]) -> int:
        """Calculate total duration from spans."""
        start_time = self._get_earliest_start_time(spans)
        end_time = self._get_latest_end_time(spans)
        if start_time and end_time:
            from datetime import datetime
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return int((end - start).total_seconds() * 1000)
        return 0
    
    def detect_framework(self, span_data: Dict[str, Any]) -> bool:
        """Detect if this is a LangChain span."""
        metadata = span_data.get('metadata', {})
        agent_name = span_data.get('agent_name', '')
        
        # Check for LangChain indicators
        return (
            'langchain' in str(metadata).lower() or
            agent_name.startswith('langchain:') or
            'stage' in metadata  # LangChain-specific metadata
        )
