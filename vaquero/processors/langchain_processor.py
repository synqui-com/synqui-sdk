"""LangChain-specific processor for hierarchical trace collection."""

from typing import Dict, List, Any, Optional
from .base_processor import FrameworkProcessor, HierarchicalTrace
from ..cost_calculator import calculate_cost

class LangChainProcessor(FrameworkProcessor):
    """LangChain processor that groups internal components under logical agents."""
    
    def __init__(self):
        self.logical_agents = {}  # logical_agent_name -> agent_data
        self.internal_components = {}  # logical_agent_name -> [components]
        self.spans = []  # All spans for this trace
        # Map span_id -> logical agent name for quick parent lookup
        self.span_to_logical: Dict[str, str] = {}
        # Buffer for orphaned internal components (tool spans that arrive before their parent)
        self.orphaned_components = []  # List of spans waiting for their parent
    
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

        else:
            # This is an internal component - find its parent logical agent
            parent_logical_agent = self._determine_parent_from_context(span_data)
            if parent_logical_agent and parent_logical_agent in self.internal_components:
                self.internal_components[parent_logical_agent].append(span_data)
                
                # Process any orphaned components that might now have a parent
                self._process_orphaned_components()
            else:
                # Buffer this component as orphaned - it will be processed later
                self.orphaned_components.append(span_data)
    
    def _process_orphaned_components(self):
        """Process orphaned components that now have a parent."""
        if not self.orphaned_components:
            return
        
        # Try to assign orphaned components to their parent
        remaining_orphans = []
        for span_data in self.orphaned_components:
            agent_name = span_data.get('agent_name', '')
            parent_logical_agent = self._determine_parent_from_context(span_data)
            
            if parent_logical_agent and parent_logical_agent in self.internal_components:
                self.internal_components[parent_logical_agent].append(span_data)
            else:
                # Still orphaned
                remaining_orphans.append(span_data)
        
        self.orphaned_components = remaining_orphans
    
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

        # NEW: For AgentExecutor, try to extract a better name from metadata
        if agent_name == 'langchain:AgentExecutor':
            # Debug: Print what we have

            # First priority: Check for agent_name in lc_meta directly (it seems to be at the top level)
            agent_name_from_config = lc_meta.get('agent_name') if lc_meta else None
            if agent_name_from_config:
                return f"langchain:{agent_name_from_config}"

            # Second priority: Try to get the actual agent name from serialized data
            serialized = lc_meta.get('serialized', {})
            if isinstance(serialized, dict):
                # Look for the actual agent name in the serialized data
                name = serialized.get('name')
                if name and name != 'AgentExecutor':
                    return f"langchain:{name}"

            # Third priority: try to get from other metadata
            chain_name = lc_meta.get('chain_name') or metadata.get('chain_name')
            if chain_name:
                return f"langchain:{chain_name}"

            # If we can't find a better name, use a generic but descriptive name
            return "langchain:AgentExecutor"

        # NEW: For other langchain chains, extract the meaningful part
        if agent_name.startswith('langchain:') and agent_name != 'langchain:chain':
            name_part = agent_name.replace('langchain:', '')
            if name_part != 'chain':
                return agent_name

        # For spans that don't match explicit patterns, they are internal components

        # For internal LangChain components, try to determine parent from context
        # This is the key intelligence - we need to group internal components
        # under their logical agents based on execution context
        return self._determine_parent_from_context(span_data)

    def _is_logical_agent_span(self, span_data: Dict[str, Any]) -> bool:
        """Determine if this span represents a logical business agent using explicit rules."""
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

        # EXCLUDE: LLM spans are internal components, not logical agents
        if agent_name.startswith('llm:'):
            return False

        # EXCLUDE: Tool spans are internal components, not logical agents
        if agent_name.startswith('tool:'):
            return False

        # EXCLUDE: Generic chain components are internal components
        if agent_name == 'langchain:chain':
            return False

        # LOGICAL AGENT: Check for explicit stage in metadata
        stage = metadata.get('stage') or lc_meta.get('stage')
        if stage:
            return True

        # LOGICAL AGENT: Check for workflow root span
        if 'workflow' in agent_name.lower():
            return True

        # LOGICAL AGENT: Check if agent_name already looks like a logical agent (ends with '_agent')
        if agent_name.endswith('_agent'):
            return True

        # LOGICAL AGENT: AgentExecutor with meaningful metadata
        if agent_name == 'langchain:AgentExecutor':
            # Only classify as logical agent if it has explicit agent_name in metadata
            agent_name_from_config = lc_meta.get('agent_name') if lc_meta else None
            if agent_name_from_config:
                return True

        # LOGICAL AGENT: Custom agent names from metadata
        if agent_name.startswith('langchain:'):
            # Check if this has an explicit agent_name in metadata
            agent_name_from_config = lc_meta.get('agent_name') if lc_meta else None
            if agent_name_from_config:
                # Only classify as logical agent if it's NOT a generic chain
                if agent_name != 'langchain:chain' and agent_name != 'langchain:AgentExecutor':
                    return True

        # All other spans are internal components
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
        # Final attempt to process any remaining orphaned components
        self._process_orphaned_components()
        
        # Log any components that are still orphaned
        if self.orphaned_components:
            for orphan in self.orphaned_components:
                pass  # Orphaned components logged but not processed
        
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

            # Get all spans for this logical agent (including internal components)
            all_spans = agent_data['spans'] + self.internal_components.get(logical_agent_name, [])
            
            # Extract model information from all spans
            model_info = self._extract_model_info(all_spans)
            
            # Extract inputs, outputs, and system prompt from spans
            # First span usually has the inputs, last span usually has the outputs
            first_span_inputs = {}
            last_span_outputs = {}
            system_prompt = None
            
            if all_spans:
                # Get inputs from the first span (usually the chain/agent start)
                first_span = all_spans[0]
                first_span_inputs = first_span.get('inputs', {})
                
                # Get outputs from the last span (usually the chain/agent end)
                last_span = all_spans[-1]
                last_span_outputs = last_span.get('outputs', {})
                
                # Try to find system prompt from any span (LLM spans usually have it)
                for span in all_spans:
                    if span.get('system_prompt'):
                        system_prompt = span.get('system_prompt')
                        break
            
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
                'start_time': self._get_earliest_start_time(all_spans),
                'end_time': self._get_latest_end_time(all_spans),
                'duration_ms': self._calculate_duration(all_spans),
                'input_tokens': sum(s.get('input_tokens', 0) for s in all_spans),
                'output_tokens': sum(s.get('output_tokens', 0) for s in all_spans),
                'total_tokens': sum(s.get('total_tokens', 0) for s in all_spans),
                'cost': model_info.get('cost', 0.0),
                'tags': {'session_id': agent_session_id} if agent_session_id else {},
                'input_data': first_span_inputs,
                'output_data': last_span_outputs,
                'metadata': {'session_id': agent_session_id} if agent_session_id else {},
                'framework_metadata': {},
                # Add model information
                'llm_model_name': model_info.get('model_name'),
                'llm_model_provider': model_info.get('model_provider'),
                'llm_model_parameters': model_info.get('model_parameters'),
                # Add system prompt
                'system_prompt': system_prompt
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

                # Extract error information from span
                span_error = span.get('error')
                error_message = None
                error_type = None
                error_stack_trace = None
                if span_error and isinstance(span_error, dict):
                    error_message = span_error.get('message')
                    error_type = span_error.get('type')
                    error_stack_trace = span_error.get('traceback')

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
                    'framework_metadata': span.get('tags', {}),
                    # Add system prompt for LLM components
                    'system_prompt': span.get('system_prompt'),
                    # Add model information for LLM components
                    'llm_model_name': span.get('model_name'),
                    'llm_model_provider': span.get('model_provider'),
                    'llm_model_parameters': span.get('model_parameters'),
                    # Add error information
                    'error': span_error,
                    'error_message': error_message,
                    'error_type': error_type,
                    'error_stack_trace': error_stack_trace
                }
                agents.append(component)
        
        # Extract metadata from spans (environment, mode, etc.)
        trace_metadata = {}
        for span in self.spans:
            if span.get('metadata'):
                # Merge metadata from spans
                trace_metadata.update(span.get('metadata', {}))
                break  # Use metadata from first span
        
        return HierarchicalTrace(
            trace_id=trace_id,
            name='workflow_root',
            agents=agents,
            dependencies=dependencies,
            metadata=trace_metadata
        )
    
    def _extract_model_info(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract model information and calculate cost from spans."""
        model_info = {
            'model_name': None,
            'model_provider': None,
            'model_parameters': None,
            'cost': 0.0
        }
        
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
