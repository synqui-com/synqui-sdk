"""LangChain-specific processor for hierarchical trace collection."""

import logging
from typing import Dict, List, Any, Optional
from .base_processor import FrameworkProcessor, HierarchicalTrace
from ..cost_calculator import calculate_cost

logger = logging.getLogger(__name__)

class LangChainProcessor(FrameworkProcessor):
    """LangChain processor that groups internal components under logical agents."""
    
    def __init__(self):
        self.logical_agents = {}  # logical_agent_name -> agent_data
        self.internal_components = {}  # logical_agent_name -> [components]
        self.spans = []  # All spans for this trace
        # Map span_id -> logical agent name for quick parent lookup
        self.span_to_logical: Dict[str, str] = {}
    
    def add_span(self, span_data: Dict[str, Any]) -> None:
        """Add LangChain span to processor."""
        self.spans.append(span_data)

        agent_name = span_data.get('agent_name', '')
        logical_agent = self._determine_logical_agent(span_data)

        # Only create logical agents for actual logical agents, not internal components
        is_logical_agent = self._is_logical_agent_span(span_data)

        if is_logical_agent:
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
            # Don't add logical agent spans to internal_components to avoid duplication

            # Record mapping for parent resolution of subsequent spans
            span_id = span_data.get('span_id')
            if span_id:
                self.span_to_logical[span_id] = logical_agent

            logger.debug(f"Added logical agent span {agent_name} to {logical_agent}")
        else:
            # This is an internal component - find its parent logical agent
            parent_logical_agent = self._determine_parent_from_context(span_data)
            if parent_logical_agent and parent_logical_agent in self.internal_components:
                self.internal_components[parent_logical_agent].append(span_data)
                logger.debug(f"Added internal component {agent_name} to {parent_logical_agent}")
            else:
                logger.debug(f"Could not determine parent for internal component {agent_name}")
    
    def _determine_logical_agent(self, span_data: Dict[str, Any]) -> str:
        """Determine which logical agent this span belongs to."""
        agent_name = span_data.get('agent_name', '')
        metadata = span_data.get('metadata', {})
        tags = span_data.get('tags', {}) or {}
        lc_meta = {}
        try:
            lc_meta_candidate = tags.get('langchain.metadata')
            if isinstance(lc_meta_candidate, dict):
                lc_meta = lc_meta_candidate
        except Exception:
            lc_meta = {}

        # Check for explicit stage in metadata - these are logical agents
        stage = metadata.get('stage') or lc_meta.get('stage')
        if stage:
            # Convert stage to agent name format (e.g., 'validation' -> 'validation_agent')
            return f"{stage}_agent"

        # Check for workflow root span - this is also a logical agent
        if 'workflow' in agent_name.lower():
            return 'workflow_root'

        # Check if agent_name already looks like a logical agent (ends with '_agent')
        if agent_name.endswith('_agent'):
            return agent_name

        # For spans that don't match explicit patterns, check if they're logical agents
        # This is a heuristic based on the span structure and metadata
        if self._is_likely_logical_agent(span_data):
            return agent_name

        # For internal LangChain components, try to determine parent from context
        # This is the key intelligence - we need to group internal components
        # under their logical agents based on execution context
        return self._determine_parent_from_context(span_data)

    def _is_logical_agent_span(self, span_data: Dict[str, Any]) -> bool:
        """Determine if this span represents a logical business agent."""
        agent_name = span_data.get('agent_name', '')
        metadata = span_data.get('metadata', {})
        tags = span_data.get('tags', {}) or {}
        lc_meta = {}
        try:
            lc_meta_candidate = tags.get('langchain.metadata')
            if isinstance(lc_meta_candidate, dict):
                lc_meta = lc_meta_candidate
        except Exception:
            lc_meta = {}

        # Check for explicit stage in metadata - these are logical agents
        stage = metadata.get('stage') or lc_meta.get('stage')
        if stage:
            return True

        # Check for workflow root span - this is also a logical agent
        if 'workflow' in agent_name.lower():
            return True

        # Check if agent_name already looks like a logical agent (ends with '_agent')
        if agent_name.endswith('_agent'):
            return True

        # Check if this is likely a logical agent based on heuristics
        if self._is_likely_logical_agent(span_data):
            return True

        # All other spans are internal components
        return False

    def _is_likely_logical_agent(self, span_data: Dict[str, Any]) -> bool:
        """Heuristic to determine if a span is likely a logical agent."""
        agent_name = span_data.get('agent_name', '')
        metadata = span_data.get('metadata', {})
        tags = span_data.get('tags', {}) or {}

        # Check if it has meaningful metadata that suggests it's a logical agent
        if metadata.get('stage') or metadata.get('agent_type') == 'logical':
            return True

        # Check if it's a root-level span with significant duration or complexity
        if (span_data.get('duration_ms', 0) > 100 or  # Significant duration
            len(tags) > 3 or  # Multiple tags suggest importance
            metadata.get('session_id') or  # Has session context
            'workflow' in agent_name.lower()):  # Workflow-related
            return True

        return False

    def _determine_parent_from_context(self, span_data: Dict[str, Any]) -> str:
        """Determine parent logical agent from execution context."""
        # Prefer explicit parent linkage when available
        parent_span_id = span_data.get('parent_span_id')
        if parent_span_id and parent_span_id in self.span_to_logical:
            return self.span_to_logical[parent_span_id]

        # Fallback heuristics - find the most appropriate parent logical agent
        agent_name = span_data.get('agent_name', '')

        if agent_name.startswith('tool:'):
            # Tool calls belong to the most recently created logical agent
            for logical_agent_name in reversed(list(self.logical_agents.keys())):
                return logical_agent_name

        elif agent_name.startswith('llm:'):
            # LLM calls belong to the most recently created logical agent
            for logical_agent_name in reversed(list(self.logical_agents.keys())):
                return logical_agent_name

        # For other internal components, use the most recently created logical agent
        if self.logical_agents:
            return list(self.logical_agents.keys())[-1]

        # Default fallback
        return 'unknown_agent'
    
    def process_trace(self, trace_id: str) -> HierarchicalTrace:
        """Process all spans into hierarchical format."""
        agents = []
        dependencies = []
        
        # Extract session_id from spans for the trace
        trace_session_id = None
        for span in self.spans:
            if span.get('metadata') and isinstance(span.get('metadata'), dict):
                trace_session_id = span['metadata'].get('session_id')
            if not trace_session_id and span.get('tags') and isinstance(span.get('tags'), dict):
                trace_session_id = span['tags'].get('session_id')
            if trace_session_id:
                break

        # Create logical agents
        for logical_agent_name, agent_data in self.logical_agents.items():
            # Extract session_id for this agent from its spans
            agent_session_id = trace_session_id

            # Extract model information from spans
            model_info = self._extract_model_info(agent_data['spans'])

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
                'cost': model_info.get('cost', 0.0),
                'tags': {'session_id': agent_session_id} if agent_session_id else {},
                'input_data': {},
                'output_data': {},
                'metadata': {'session_id': agent_session_id} if agent_session_id else {},
                'framework_metadata': {},
                # Add model information
                'llm_model_name': model_info.get('model_name'),
                'llm_model_provider': model_info.get('model_provider'),
                'llm_model_parameters': model_info.get('model_parameters')
            }
            agents.append(logical_agent)

            # Create internal components for this logical agent
            components = self.internal_components.get(logical_agent_name, [])
            for span in components:
                # Preserve session_id in component metadata
                component_metadata = span.get('metadata', {}).copy()
                component_tags = span.get('tags', {}).copy()
                if agent_session_id:
                    component_metadata['session_id'] = agent_session_id
                    component_tags['session_id'] = agent_session_id

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
                    'tags': component_tags,
                    'input_data': span.get('inputs', {}),
                    'output_data': span.get('outputs', {}),
                    'metadata': component_metadata,
                    'framework_metadata': span.get('tags', {})
                }
                agents.append(component)
        
        return HierarchicalTrace(
            trace_id=trace_id,
            name='workflow_root',
            agents=agents,
            dependencies=dependencies
        )
    
    def _extract_model_info(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract model information and calculate cost from spans."""
        model_info = {
            'model_name': None,
            'model_provider': None,
            'model_parameters': None,
            'cost': 0.0
        }
        
        # Debug: Log what's in the spans
        logger.debug(f"Extracting model info from {len(spans)} spans")
        for i, span in enumerate(spans):
            logger.debug(f"Span {i}: {span.get('agent_name', 'unknown')} - model_name: {span.get('model_name')}, model_provider: {span.get('model_provider')}")
        
        # Look for model information in spans
        for span in spans:
            # Check if span has model information
            if span.get('model_name'):
                model_info['model_name'] = span.get('model_name')
                model_info['model_provider'] = span.get('model_provider')
                model_info['model_parameters'] = span.get('model_parameters')
                
                # Calculate cost based on token usage
                input_tokens = span.get('input_tokens', 0)
                output_tokens = span.get('output_tokens', 0)
                if input_tokens > 0 or output_tokens > 0:
                    cost = calculate_cost(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model_name=model_info['model_name'],
                        provider=model_info['model_provider']
                    )
                    model_info['cost'] = cost
                    logger.debug(f"Calculated cost: ${cost:.6f} for {input_tokens} input + {output_tokens} output tokens")
                
                logger.debug(f"Found model info: {model_info}")
                break  # Use the first model info found
        
        return model_info

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
