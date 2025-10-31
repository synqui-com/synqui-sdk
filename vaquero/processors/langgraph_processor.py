"""LangGraph-specific processor for hierarchical trace collection."""

import logging
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

        # Group spans by type
        graph_spans = [s for s in self.spans if s.get('component_type') == 'graph']
        node_spans = [s for s in self.spans if s.get('component_type') == 'node']
        llm_spans = [s for s in self.spans if s.get('component_type') == 'llm']
        tool_spans = [s for s in self.spans if s.get('component_type') == 'tool']
        chain_spans = [s for s in self.spans if s.get('component_type') == 'chain']

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

        hierarchical_trace = HierarchicalTrace(
            trace_id=trace_id,
            name=f"LangGraph Workflow" if not chat_session_id else f"Chat Session {chat_session_id[:8]}",
            agents=agents,
            dependencies=dependencies,
            metadata={
                'framework': 'langgraph',
                'chat_session_id': chat_session_id,
                'total_spans': len(self.spans),
                'graph_spans': len(graph_spans),
                'node_spans': len(node_spans),
                'llm_spans': len(llm_spans),
                'tool_spans': len(tool_spans),
                'chain_spans': len(chain_spans)
            }
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
        nodes_by_sequence = {}
        for span in node_spans:
            seq = span.get('message_sequence', 0)
            if seq not in nodes_by_sequence:
                nodes_by_sequence[seq] = []
            nodes_by_sequence[seq].append(span)

        # Create orchestration agents (Level 2)
        for seq, node_list in nodes_by_sequence.items():
            orchestration_agent = self._build_orchestration_agent(seq, node_list, llm_spans, tool_spans, chain_spans)
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
            orchestration_agent['agents'].append(node_agent)

        return orchestration_agent

    def _build_node_agent(self, node_span: Dict[str, Any], llm_spans: List[Dict],
                         tool_spans: List[Dict], chain_spans: List[Dict]) -> Dict[str, Any]:
        """Build a node agent (Level 3) with its component spans."""

        node_name = node_span.get('name', 'unknown_node')

        # Find components that belong to this node (by parent_span_id)
        node_span_id = node_span.get('span_id')
        node_components = []

        # Add LLM components
        for llm_span in llm_spans:
            if llm_span.get('parent_span_id') == node_span_id:
                node_components.append(self._span_to_agent(llm_span))

        # Add tool components
        for tool_span in tool_spans:
            if tool_span.get('parent_span_id') == node_span_id:
                node_components.append(self._span_to_agent(tool_span))

        # Add chain components
        for chain_span in chain_spans:
            if chain_span.get('parent_span_id') == node_span_id:
                node_components.append(self._span_to_agent(chain_span))

        # Aggregate metrics from components
        total_tokens = sum(c.get('total_tokens', 0) for c in node_components)
        total_cost = sum(c.get('total_cost', 0) for c in node_components)

        node_agent = {
            'name': node_name,
            'level': 3,
                    'framework': 'langgraph',
                    'component_type': 'agent',
            'parent_agent_id': None,  # Will be set when added to orchestration
            'start_time': node_span.get('start_time'),
            'end_time': node_span.get('end_time'),
            'status': node_span.get('status', 'completed'),
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
            'agents': node_components  # Components become sub-agents
        }

        # Extract model info from LLM components
        model_info = self._extract_model_info([c for c in node_components if c.get('component_type') == 'llm'])
        if model_info:
            node_agent['model_info'] = model_info

        return node_agent

    def _span_to_agent(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a normalized span to agent format."""
        agent = {
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
            'agents': []  # Components don't have sub-components
        }

        # Add optional fields
        if span.get('system_prompt'):
            agent['system_prompt'] = span['system_prompt']
        if span.get('llm_model_name'):
            agent['llm_model_name'] = span['llm_model_name']
        if span.get('llm_model_provider'):
            agent['llm_model_provider'] = span['llm_model_provider']
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
