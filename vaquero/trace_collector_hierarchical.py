"""
Hierarchical trace collector for LangChain integration.

This collector implements the hierarchical storage approach where:
- Logical agents (validation, analysis, report) are stored as level 1 agents
- Internal LangChain components are stored as level 2+ children
- All components are grouped under a single trace
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class HierarchicalTraceCollector:
    """Hierarchical trace collector that groups spans by logical agents."""
    
    def __init__(self, sdk: 'VaqueroSDK'):
        """Initialize the hierarchical trace collector.
        
        Args:
            sdk: Reference to the main SDK instance
        """
        self.sdk = sdk
        self.config = sdk.config
        self._traces: Dict[str, Dict[str, Any]] = {}  # trace_id -> trace data
        self._logical_agents: Dict[str, Dict[str, Any]] = {}  # agent_name -> agent data
        self._internal_components: Dict[str, List[Dict[str, Any]]] = {}  # agent_name -> components
        self._span_to_agent: Dict[str, str] = {}  # span_id -> logical_agent_name
    
    def process_span(self, span_data: Dict[str, Any]) -> None:
        """Process a span and determine which logical agent it belongs to.

        Args:
            span_data: Span data to process
        """
        span_id = span_data.get('span_id')
        parent_span_id = span_data.get('parent_span_id')
        session_id = span_data.get('session_id')
        metadata = span_data.get('metadata', {})
        agent_name = span_data.get('agent_name', 'unknown')
        
        # Get session_id from metadata if not in span_data
        if not session_id and metadata:
            session_id = metadata.get('session_id')
        
        # Determine which logical agent this span belongs to
        logical_agent_name = self._determine_logical_agent(span_data, session_id)
        
        # Store the mapping
        self._span_to_agent[span_id] = logical_agent_name
        
        # Create or update logical agent
        if logical_agent_name not in self._logical_agents:
            self._logical_agents[logical_agent_name] = self._create_logical_agent(
                logical_agent_name, session_id
            )
        
        # Add internal component
        if logical_agent_name not in self._internal_components:
            self._internal_components[logical_agent_name] = []
        
        component = self._create_internal_component(span_data, logical_agent_name)
        self._internal_components[logical_agent_name].append(component)
        
        # Update logical agent metrics
        self._update_logical_agent_metrics(logical_agent_name, span_data)
        
        logger.debug(f"Added {agent_name} to logical agent {logical_agent_name}")
    
    def _determine_logical_agent(self, span_data: Dict[str, Any], session_id: str) -> str:
        """Determine which logical agent a span belongs to."""
        agent_name = span_data.get('agent_name', '')
        metadata = span_data.get('metadata', {})
        
        # Check if this is a root workflow span
        if 'workflow' in agent_name.lower():
            return 'workflow_root'
        
        # Check metadata for stage information
        stage = metadata.get('stage')
        if stage:
            return f"{stage}_agent"
        
        # Check if this is a tool span
        if agent_name.startswith('tool:'):
            # Try to determine which logical agent this tool belongs to
            # by looking at the parent span's logical agent
            parent_span_id = span_data.get('parent_span_id')
            if parent_span_id and parent_span_id in self._span_to_agent:
                return self._span_to_agent[parent_span_id]
            
            # Default tool grouping based on tool name - use generic approach
            tool_name = agent_name.replace('tool:', '')
            # Extract the base function name from the tool name
            base_name = tool_name.split('_')[0] if '_' in tool_name else tool_name
            return f"{base_name}_agent"
        
        # Check if this is an LLM span
        if agent_name.startswith('llm:'):
            # Try to determine which logical agent this LLM belongs to
            parent_span_id = span_data.get('parent_span_id')
            if parent_span_id and parent_span_id in self._span_to_agent:
                return self._span_to_agent[parent_span_id]
        
        # Check if this is a chain span
        if agent_name.startswith('langchain:chain'):
            # Try to determine which logical agent this chain belongs to
            parent_span_id = span_data.get('parent_span_id')
            if parent_span_id and parent_span_id in self._span_to_agent:
                return self._span_to_agent[parent_span_id]
        
        # Default grouping based on session_id and agent patterns
        if session_id:
            # Group by session_id and try to infer logical agent from context
            return 'unknown_agent'
        
        return 'unknown_agent'
    
    def _create_logical_agent(self, agent_name: str, session_id: str) -> Dict[str, Any]:
        """Create a logical agent record."""
        return {
            'id': str(uuid.uuid4()),
            'name': agent_name,
            'type': 'agent',
            'framework': 'langchain',
            'level': 1,
            'parent_agent_id': None,
            'session_id': session_id,
            'status': 'running',
            'start_time': datetime.utcnow().isoformat(),
            'end_time': None,
            'duration_ms': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'success_rate': 1.0,
            'error_count': 0,
            'component_count': 0,
            'framework_metadata': {
                'session_id': session_id,
                'agent_type': 'logical'
            }
        }
    
    def _create_internal_component(self, span_data: Dict[str, Any], logical_agent_name: str) -> Dict[str, Any]:
        """Create an internal component record."""
        agent_name = span_data.get('agent_name', 'unknown')
        
        # Determine component type
        component_type = self._map_component_type(agent_name)
        
        # Determine level based on component type
        level = self._calculate_level(agent_name)
        
        return {
            'id': str(uuid.uuid4()),
            'name': agent_name,
            'type': component_type,
            'framework': 'langchain',
            'level': level,
            'parent_agent_id': self._logical_agents[logical_agent_name]['id'],
            'span_id': span_data.get('span_id'),
            'parent_span_id': span_data.get('parent_span_id'),
            'status': span_data.get('status', 'running'),
            'start_time': span_data.get('start_time'),
            'end_time': span_data.get('end_time'),
            'duration_ms': span_data.get('duration_ms', 0),
            'tokens': span_data.get('tokens', 0),
            'cost': span_data.get('cost', 0.0),
            'framework_metadata': {
                'session_id': span_data.get('session_id'),
                'component_type': component_type,
                'original_span_data': span_data
            }
        }
    
    def _map_component_type(self, agent_name: str) -> str:
        """Map agent name to component type."""
        if agent_name.startswith('tool:'):
            return 'tool'
        elif agent_name.startswith('llm:'):
            return 'llm'
        elif agent_name.startswith('langchain:chain'):
            return 'chain'
        elif agent_name.startswith('langchain:ChatPromptTemplate'):
            return 'prompt'
        elif agent_name.startswith('langchain:'):
            return 'langchain_component'
        else:
            return 'unknown'
    
    def _calculate_level(self, agent_name: str) -> int:
        """Calculate the hierarchical level of a component."""
        if agent_name.startswith('tool:'):
            return 3  # Tools are deepest level
        elif agent_name.startswith('llm:'):
            return 3  # LLM calls are deepest level
        elif agent_name.startswith('langchain:ChatPromptTemplate'):
            return 3  # Prompt templates are deepest level
        elif agent_name.startswith('langchain:chain'):
            return 2  # Chains are intermediate level
        elif agent_name.startswith('langchain:'):
            return 2  # Other LangChain components are intermediate level
        else:
            return 2  # Default intermediate level
    
    def _update_logical_agent_metrics(self, logical_agent_name: str, span_data: Dict[str, Any]):
        """Update logical agent metrics based on new span data."""
        agent = self._logical_agents[logical_agent_name]
        
        # Update component count
        agent['component_count'] = len(self._internal_components[logical_agent_name])
        
        # Update duration (use the latest end time)
        end_time = span_data.get('end_time')
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                start_dt = datetime.fromisoformat(agent['start_time'].replace('Z', '+00:00'))
                duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
                agent['duration_ms'] = max(agent['duration_ms'], duration_ms)
            except:
                pass
        
        # Update tokens and cost
        agent['total_tokens'] += span_data.get('tokens', 0)
        agent['total_cost'] += span_data.get('cost', 0.0)
        
        # Update status
        if span_data.get('status') == 'error':
            agent['error_count'] += 1
            agent['success_rate'] = 1.0 - (agent['error_count'] / agent['component_count'])
    
    def finalize_trace(self, trace_id: str) -> Dict[str, Any]:
        """Finalize a trace and return the hierarchical structure."""
        # Create the final trace structure
        trace_data = {
            'trace_id': trace_id,
            'name': 'workflow_root',
            'status': 'completed',
            'session_id': None,
            'start_time': None,
            'end_time': None,
            'duration_ms': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'logical_agents': [],
            'agent_count': len(self._logical_agents),
            'total_components': sum(len(components) for components in self._internal_components.values())
        }
        
        # Add logical agents with their components
        for agent_name, agent_data in self._logical_agents.items():
            # Finalize agent
            agent_data['status'] = 'completed'
            agent_data['end_time'] = datetime.utcnow().isoformat()
            
            # Add components
            agent_data['components'] = self._internal_components.get(agent_name, [])
            
            # Update trace metrics
            trace_data['total_tokens'] += agent_data['total_tokens']
            trace_data['total_cost'] += agent_data['total_cost']
            trace_data['duration_ms'] = max(trace_data['duration_ms'], agent_data['duration_ms'])
            
            trace_data['logical_agents'].append(agent_data)
        
        # Set trace start/end times
        if trace_data['logical_agents']:
            start_times = [agent['start_time'] for agent in trace_data['logical_agents'] if agent['start_time']]
            end_times = [agent['end_time'] for agent in trace_data['logical_agents'] if agent['end_time']]
            
            if start_times:
                trace_data['start_time'] = min(start_times)
            if end_times:
                trace_data['end_time'] = max(end_times)
        
        return trace_data
    
    def end_trace(self, trace_id: str, end_data: Dict[str, Any]) -> None:
        """End a trace and send the hierarchical structure to the batch processor."""
        try:
            # Finalize the trace structure
            trace_data = self.finalize_trace(trace_id)
            
            # Send to batch processor
            if self.sdk._batch_processor:
                self.sdk._batch_processor.add_trace(trace_data)
                logger.info(f"Hierarchical trace sent: {trace_data['agent_count']} logical agents, {trace_data['total_components']} components")
            else:
                logger.warning("No batch processor available to send hierarchical trace")
                
        except Exception as e:
            logger.error(f"Failed to end hierarchical trace: {e}", exc_info=True)
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get a summary of the current trace state."""
        return {
            'logical_agents': len(self._logical_agents),
            'total_components': sum(len(components) for components in self._internal_components.values()),
            'agents': list(self._logical_agents.keys()),
            'component_counts': {
                agent: len(components) 
                for agent, components in self._internal_components.items()
            }
        }
