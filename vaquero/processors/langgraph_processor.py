"""LangGraph-specific processor for hierarchical trace collection."""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_processor import FrameworkProcessor, HierarchicalTrace

logger = logging.getLogger(__name__)

class LangGraphProcessor(FrameworkProcessor):
    """LangGraph processor that builds hierarchical traces from normalized spans."""
    
    def __init__(self):
        self.spans = []  # All normalized spans for this trace

    def add_span(self, span_data: Dict[str, Any]) -> None:
        """Add normalized LangGraph span to processor."""
        self.spans.append(span_data)
        logger.debug(f"Added LangGraph span: {span_data.get('component_type')} {span_data.get('name')}")
    
    def process_trace(self, trace_id: str) -> HierarchicalTrace:
        """Process normalized spans into hierarchical format matching CHAT_SESSION_TRACE_HIERARCHY."""
        logger.info(f"Processing LangGraph trace {trace_id} with {len(self.spans)} spans")

        # Log all spans received
        logger.info(f"📊 PROCESSOR: All spans received:")
        for i, span in enumerate(self.spans):
            logger.info(f"📊 PROCESSOR: Span {i}: component_type={span.get('component_type')}, "
                       f"name={span.get('name')}, span_id={span.get('span_id')}, "
                       f"message_sequence={span.get('message_sequence')}")

        # Group spans by type
        graph_spans = [s for s in self.spans if s.get('component_type') == 'graph']
        node_spans = [s for s in self.spans if s.get('component_type') == 'node']
        llm_spans = [s for s in self.spans if s.get('component_type') == 'llm']
        tool_spans = [s for s in self.spans if s.get('component_type') == 'tool']
        chain_spans = [s for s in self.spans if s.get('component_type') == 'chain']
        
        logger.info(f"📊 PROCESSOR: Grouped spans - graph={len(graph_spans)}, node={len(node_spans)}, "
                   f"llm={len(llm_spans)}, tool={len(tool_spans)}, chain={len(chain_spans)}")

        # Extract session info
        chat_session_id = None
        for span in self.spans:
            if span.get('chat_session_id'):
                chat_session_id = span['chat_session_id']
                break

        agents = []
        dependencies = []

        if chat_session_id:
            # Build session hierarchy: Session → Orchestration → Agent → Components
            session_agent = self._build_session_hierarchy(trace_id, chat_session_id, graph_spans, node_spans, llm_spans, tool_spans, chain_spans)
            agents.append(session_agent)
        else:
            # Fallback for non-session traces
            for span in self.spans:
                if span.get('component_type') in ['node', 'llm', 'tool', 'chain']:
                    agent = self._span_to_agent(span)
                    agents.append(agent)

        # Extract graph architecture from any span metadata if available
        # (It may be in graph spans, or in any span as a fallback if on_graph_start wasn't called)
        graph_architecture = None
        # First check graph spans
        for span in graph_spans:
            if span.get('metadata', {}).get('graph_architecture'):
                graph_architecture = span['metadata']['graph_architecture']
                logger.info(f"📊 PROCESSOR: Found graph architecture in graph span: {len(graph_architecture.get('nodes', []))} nodes")
                break
        # Fallback: check any span if no graph spans exist
        if not graph_architecture:
            for span in self.spans:
                if span.get('metadata', {}).get('graph_architecture'):
                    graph_architecture = span['metadata']['graph_architecture']
                    logger.info(f"📊 PROCESSOR: Found graph architecture in span metadata (fallback): {len(graph_architecture.get('nodes', []))} nodes")
                    break
        
        metadata = {
            'framework': 'langgraph',
            'chat_session_id': chat_session_id,
            'total_spans': len(self.spans),
            'graph_spans': len(graph_spans),
            'node_spans': len(node_spans),
            'llm_spans': len(llm_spans),
            'tool_spans': len(tool_spans),
            'chain_spans': len(chain_spans)
        }
        
        # Add graph architecture to metadata if found
        if graph_architecture:
            metadata['graph_architecture'] = graph_architecture
        
        hierarchical_trace = HierarchicalTrace(
            trace_id=trace_id,
            name=f"LangGraph Workflow" if not chat_session_id else f"Chat Session {chat_session_id[:8]}",
            agents=agents,
            dependencies=dependencies,
            metadata=metadata
        )

        logger.info(f"Built hierarchical trace with {len(agents)} top-level agents")
        return hierarchical_trace

    def _build_session_hierarchy(self, trace_id: str, chat_session_id: str, graph_spans: List[Dict],
                                node_spans: List[Dict], llm_spans: List[Dict], tool_spans: List[Dict],
                                chain_spans: List[Dict]) -> Dict[str, Any]:
        """Build the session hierarchy: Session → Orchestration → Agent → Components."""

        # Level 1: Session orchestration
        session_start = self._get_earliest_start_time(self.spans)
        session_end = self._get_latest_end_time(self.spans)

        session_agent = {
                'name': f"chat_session_{chat_session_id[:8]}",
                'level': 1,
                'framework': 'langgraph',
                'component_type': 'session_orchestration',
                'parent_agent_id': None,
                'chat_session_id': chat_session_id,
            'start_time': session_start,
            'end_time': session_end,
                'status': 'completed',
                'agents': []
            }
            
        # Group nodes by message sequence for orchestrations (Level 2)
        logger.info(f"📊 PROCESSOR: Grouping {len(node_spans)} node spans by message_sequence")
        nodes_by_sequence = {}
        for span in node_spans:
            seq = span.get('message_sequence', 0)
            logger.info(f"📊 PROCESSOR: Node span - name={span.get('name')}, message_sequence={seq}")
            if seq not in nodes_by_sequence:
                nodes_by_sequence[seq] = []
            nodes_by_sequence[seq].append(span)

        logger.info(f"📊 PROCESSOR: Created {len(nodes_by_sequence)} message sequence groups: {list(nodes_by_sequence.keys())}")

        # Create orchestration agents (Level 2)
        for seq, node_list in nodes_by_sequence.items():
            logger.info(f"📊 PROCESSOR: Creating orchestration for sequence {seq} with {len(node_list)} nodes")
            orchestration_agent = self._build_orchestration_agent(seq, node_list, llm_spans, tool_spans, chain_spans)
            # Ensure proper parent linkage in flattened output
            orchestration_agent['parent_agent_id'] = session_agent['name']
            session_agent['agents'].append(orchestration_agent)

        return session_agent

    def _build_orchestration_agent(self, sequence: int, node_spans: List[Dict], llm_spans: List[Dict],
                                  tool_spans: List[Dict], chain_spans: List[Dict]) -> Dict[str, Any]:
        """Build an orchestration agent (Level 2) containing multiple node agents."""

        orchestration_start = self._get_earliest_start_time(node_spans)
        orchestration_end = self._get_latest_end_time(node_spans)

        orchestration_agent = {
            'name': f"agent_orchestration_{sequence}",
                    'level': 2,
                    'framework': 'langgraph',
                    'component_type': 'agent_orchestration',
            'parent_agent_id': None,  # Will be set when added to session
            'message_sequence': sequence,
            'start_time': orchestration_start,
            'end_time': orchestration_end,
                    'status': 'completed',
                    'agents': []
                }
                
        # Add node agents (Level 3)
        for node_span in node_spans:
            node_agent = self._build_node_agent(node_span, llm_spans, tool_spans, chain_spans)
            # Ensure node shows as child of this orchestration in flattened DB rows
            node_agent['parent_agent_id'] = orchestration_agent['name']
            orchestration_agent['agents'].append(node_agent)

        return orchestration_agent

    def _find_components_for_node(self, node_span_id: str, all_spans: List[Dict]) -> List[Dict]:
        """Find all component spans that belong to a node by traversing parent chains."""
        components = []

        for span in all_spans:
            current_span = span
            visited = set()  # Prevent cycles

            # Walk up the parent chain to see if we reach the target node
            while current_span:
                current_parent_id = current_span.get('parent_span_id')
                if not current_parent_id or current_parent_id in visited:
                    break

                visited.add(current_parent_id)

                # Check if this parent is our target node
                if current_parent_id == node_span_id:
                    components.append(span)
                    break

                # Find the parent span and continue walking
                current_span = next((s for s in all_spans if s.get('span_id') == current_parent_id), None)

        return components

    def _build_node_agent(self, node_span: Dict[str, Any], llm_spans: List[Dict],
                         tool_spans: List[Dict], chain_spans: List[Dict]) -> Dict[str, Any]:
        """Build a node agent (Level 3) with its component spans."""

        node_name = node_span.get('name', 'unknown_node')

        # Find components that belong to this node using parent chain traversal
        node_span_id = node_span.get('span_id')
        all_component_spans = llm_spans + tool_spans + chain_spans
        component_spans = self._find_components_for_node(node_span_id, all_component_spans)
        node_components = [self._span_to_agent(span) for span in component_spans]

        # Propagate parent linkage so DB rows reflect association
        for component in node_components:
            component['parent_agent_id'] = node_name

        # Aggregate metrics from components
        total_tokens = sum(c.get('total_tokens', 0) for c in node_components)
        input_tokens = sum(c.get('input_tokens', 0) for c in node_components)
        output_tokens = sum(c.get('output_tokens', 0) for c in node_components)
        total_cost = sum(c.get('cost', 0.0) for c in node_components)
        
        # Calculate duration from component spans (or use node span if available)
        component_start = self._get_earliest_start_time(component_spans)
        component_end = self._get_latest_end_time(component_spans)
        node_start = node_span.get('start_time') or component_start
        node_end = node_span.get('end_time') or component_end
        
        # Calculate duration_ms
        duration_ms = 0
        if node_start and node_end:
            try:
                start_dt = datetime.fromisoformat(node_start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(node_end.replace('Z', '+00:00'))
                duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
            except (ValueError, AttributeError):
                # Fallback: calculate from components
                if component_start and component_end:
                    try:
                        start_dt = datetime.fromisoformat(component_start.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(component_end.replace('Z', '+00:00'))
                        duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
                    except (ValueError, AttributeError):
                        pass
        
        # Extract system prompt from first LLM component
        llm_components = [c for c in node_components if c.get('component_type') == 'llm']
        system_prompt = None
        for llm_comp in llm_components:
            if llm_comp.get('system_prompt'):
                system_prompt = llm_comp.get('system_prompt')
                break
        
        # Extract model information from LLM components
        llm_model_name = None
        llm_model_provider = None
        llm_model_parameters = None
        for llm_comp in llm_components:
            if llm_comp.get('llm_model_name') and not llm_model_name:
                llm_model_name = llm_comp.get('llm_model_name')
            if llm_comp.get('llm_model_provider') and not llm_model_provider:
                llm_model_provider = llm_comp.get('llm_model_provider')
            if llm_comp.get('llm_model_parameters') and not llm_model_parameters:
                llm_model_parameters = llm_comp.get('llm_model_parameters')
        
        # Aggregate input_data and output_data from components
        # Merge all input_data and output_data from child components
        aggregated_input_data = {}
        aggregated_output_data = {}
        for component in node_components:
            if component.get('input_data'):
                if isinstance(component['input_data'], dict):
                    aggregated_input_data.update(component['input_data'])
            if component.get('output_data'):
                if isinstance(component['output_data'], dict):
                    aggregated_output_data.update(component['output_data'])

        node_agent = {
            'name': node_name,
            'level': 3,
                    'framework': 'langgraph',
                    'component_type': 'agent',
            'parent_agent_id': None,  # Will be set when added to orchestration
            'start_time': node_start,
            'end_time': node_end,
            'duration_ms': duration_ms,
            'status': node_span.get('status', 'completed'),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
            'cost': total_cost,
            'agents': node_components,  # Components become sub-agents
            'input_data': aggregated_input_data if aggregated_input_data else None,
            'output_data': aggregated_output_data if aggregated_output_data else None
        }
        
        # Add system prompt if found
        if system_prompt:
            node_agent['system_prompt'] = system_prompt
        
        # Add model information if found
        if llm_model_name:
            node_agent['llm_model_name'] = llm_model_name
        if llm_model_provider:
            node_agent['llm_model_provider'] = llm_model_provider
        if llm_model_parameters:
            node_agent['llm_model_parameters'] = llm_model_parameters

        # Extract model info from LLM components (for backward compatibility)
        model_info = self._extract_model_info(llm_components)
        if model_info:
            node_agent['model_info'] = model_info

        return node_agent

    def _span_to_agent(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a normalized span to agent format."""
        agent = {
            'agent_id': str(uuid.uuid4()),  # Generate UUID for consistency
            'name': span.get('name', 'unknown'),
            'level': self._component_type_to_level(span.get('component_type', 'unknown')),
                'framework': 'langgraph',
            'component_type': span.get('component_type', 'unknown'),
            'parent_agent_id': None,  # Will be set by parent
            'start_time': span.get('start_time'),
            'end_time': span.get('end_time'),
            'status': span.get('status', 'completed'),
            'input_tokens': span.get('input_tokens', 0),
            'output_tokens': span.get('output_tokens', 0),
            'total_tokens': span.get('total_tokens', 0),
            'cost': span.get('cost', 0.0),
            'agents': [],  # Components don't have sub-components
            # Extract input/output data from span
            'input_data': span.get('inputs', {}),
            'output_data': span.get('outputs', {})
        }

        # Add optional fields
        if span.get('system_prompt'):
            agent['system_prompt'] = span['system_prompt']
        if span.get('llm_model_name'):
            agent['llm_model_name'] = span['llm_model_name']
        if span.get('llm_model_provider'):
            agent['llm_model_provider'] = span['llm_model_provider']
        if span.get('llm_model_parameters'):
            agent['llm_model_parameters'] = span['llm_model_parameters']
        if span.get('prompt_hash'):
            agent['prompt_hash'] = span['prompt_hash']
        if span.get('prompt_name'):
            agent['prompt_name'] = span['prompt_name']
        if span.get('prompt_version'):
            agent['prompt_version'] = span['prompt_version']
        if span.get('reasoning'):
            agent['reasoning'] = span['reasoning']

        return agent

    def _component_type_to_level(self, component_type: str) -> int:
        """Map component type to hierarchy level."""
        level_map = {
            'graph': 1,
            'session_orchestration': 1,
            'agent_orchestration': 2,
            'node': 3,
            'agent': 3,
            'llm': 4,
            'tool': 4,
            'chain': 4,
            'prompt': 4
        }
        return level_map.get(component_type, 3)

    def _extract_model_info(self, llm_agents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract model information from LLM agents."""
        for agent in llm_agents:
            model_info = {}
            if agent.get('llm_model_name'):
                model_info['model_name'] = agent['llm_model_name']
            if agent.get('llm_model_provider'):
                model_info['model_provider'] = agent['llm_model_provider']
            if agent.get('llm_model_parameters'):
                model_info['model_parameters'] = agent['llm_model_parameters']
            if model_info:
                return model_info
        return None
    
    def _get_earliest_start_time(self, spans: List[Dict[str, Any]]) -> Optional[str]:
        """Get earliest start time from spans."""
        start_times = [s.get('start_time') for s in spans if s.get('start_time')]
        return min(start_times) if start_times else None
    
    def _get_latest_end_time(self, spans: List[Dict[str, Any]]) -> Optional[str]:
        """Get latest end time from spans."""
        end_times = [s.get('end_time') for s in spans if s.get('end_time')]
        return max(end_times) if end_times else None

    def detect_framework(self, span_data: Dict[str, Any]) -> bool:
        """Detect if this is a LangGraph span."""
        return (
            span_data.get('component_type') in ['graph', 'node', 'llm', 'tool', 'chain'] or
            span_data.get('chat_session_id') is not None
        )
