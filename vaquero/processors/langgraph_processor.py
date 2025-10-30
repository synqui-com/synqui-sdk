"""LangGraph-specific processor for hierarchical trace collection."""

import logging
from typing import Dict, List, Any, Optional
from .base_processor import FrameworkProcessor, HierarchicalTrace
from ..cost_calculator import calculate_cost

logger = logging.getLogger(__name__)

class LangGraphProcessor(FrameworkProcessor):
    """LangGraph processor that handles agent swarm workflows."""
    
    def __init__(self):
        self.agents = {}  # agent_name -> agent_data
        self.spans = []  # All spans for this trace
        # Ensure orchestration container is present for grouping
        self.agent_orchestrations = getattr(self, 'agent_orchestrations', {})
        self.agent_orchestrations = {}  # orchestration_id -> orchestration_data
    
    def add_span(self, span_data: Dict[str, Any]) -> None:
        """Add LangGraph span to processor."""
        self.spans.append(span_data)

        agent_name = span_data.get('agent_name', '')
        component_type = span_data.get('component_type', 'agent')

        # 🔍 Enhanced logging for debugging
        agent_orchestration_id = span_data.get('agent_orchestration_id')
        session_id = span_data.get('session_id', 'none')
        logger.info(f"📥 LANGGRAPH PROCESSOR: Adding span - agent: {agent_name}, type: {component_type}, orchestration: {agent_orchestration_id or 'none'}")
        logger.info(f"📥 LANGGRAPH PROCESSOR: Session: {session_id}, total spans now: {len(self.spans)}")

        logger.debug(f"🔍 Span data keys: {list(span_data.keys())}")
        logger.debug(f"🔍 Span duration: {span_data.get('duration', 'N/A')}")
        logger.debug(f"🔍 Span tokens - input: {span_data.get('input_tokens', 0)}, output: {span_data.get('output_tokens', 0)}")
        logger.debug(f"🔍 Span cost: {span_data.get('cost', 0.0)}")
        logger.debug(f"🔍 Span status: {span_data.get('status', 'unknown')}")
        logger.debug(f"🔍 Span start_time: {span_data.get('start_time', 'N/A')}")
        logger.debug(f"🔍 Span end_time: {span_data.get('end_time', 'N/A')}")

        # Extract session information
        chat_session_id = span_data.get('chat_session_id')

        logger.debug(f"🔍 Session info - session_id: {session_id}, chat_session_id: {chat_session_id}, orchestration_id: {agent_orchestration_id}")

        # Create or update agent orchestration
        if agent_orchestration_id and agent_orchestration_id not in self.agent_orchestrations:
            self.agent_orchestrations[agent_orchestration_id] = {
                'id': agent_orchestration_id,
                'session_id': session_id,
                'chat_session_id': chat_session_id,
                'agents': [],
                'components': [],
                'start_time': span_data.get('start_time'),
                'end_time': span_data.get('end_time'),
                'status': span_data.get('status', 'completed')
            }
        
        # Determine the level based on component type
        if component_type == 'agent':
            level = 2  # Individual agents are level 2
        elif component_type in ['llm', 'tool', 'chain', 'prompt']:
            level = 3  # Internal components are level 3
        else:
            level = 2  # Default to level 2
        
        # Add span to orchestration bucket (agents vs components)
        if agent_orchestration_id:
            # For internal components (llm/tool/chain/prompt), attach to components list
            if component_type in ['llm', 'tool', 'chain', 'prompt']:
                parent_agent_id = span_data.get('parent_agent_name') or self._find_parent_agent_for_span(span_data)
                component_record = {
                    'name': agent_name,
                    'level': 4,  # Components are level 4
                    'framework': 'langgraph',
                    'component_type': component_type,
                    'parent_agent_id': parent_agent_id,
                    'spans': [span_data],
                    'session_id': session_id,
                    'chat_session_id': chat_session_id,
                    'agent_orchestration_id': agent_orchestration_id,
                }
                self.agent_orchestrations[agent_orchestration_id]['components'].append(component_record)
                logger.debug(f"Added LangGraph component {component_type} {agent_name} under parent {parent_agent_id} to orchestration {agent_orchestration_id}")
            else:
                agent_data = {
                    'name': agent_name,
                    'level': level,
                    'framework': 'langgraph',
                    'component_type': component_type,
                    'parent_agent_id': None,
                    'spans': [span_data],
                    'session_id': session_id,
                    'chat_session_id': chat_session_id,
                    'agent_orchestration_id': agent_orchestration_id,
                    'message_type': span_data.get('message_type', 'agent_response'),
                    'message_sequence': span_data.get('message_sequence', 0)
                }
                self.agent_orchestrations[agent_orchestration_id]['agents'].append(agent_data)
                logger.debug(f"Added LangGraph agent {agent_name} to orchestration {agent_orchestration_id}")
        else:
            # Standalone agent/component
            if agent_name not in self.agents:
                self.agents[agent_name] = {
                    'name': agent_name,
                    'level': level,
                    'framework': 'langgraph',
                    'component_type': component_type,
                    'parent_agent_id': None,
                    'spans': []
                }
            
            self.agents[agent_name]['spans'].append(span_data)
            logger.debug(f"Added standalone LangGraph {component_type} {agent_name}")
    
    def process_trace(self, trace_id: str) -> HierarchicalTrace:
        """Process all spans into hierarchical format."""
        agents = []
        dependencies = []

        # 🔍 DEBUG LOGGING: Show what data we have
        logger.info(f"🔍 PROCESSOR DEBUG: Processing trace {trace_id}")
        logger.info(f"🔍 PROCESSOR DEBUG: Total spans: {len(self.spans)}")
        logger.info(f"🔍 PROCESSOR DEBUG: Standalone agents: {list(self.agents.keys())}")
        logger.info(f"🔍 PROCESSOR DEBUG: Agent orchestrations: {list(self.agent_orchestrations.keys())}")

        # Show details of orchestrations
        for orch_id, orch_data in self.agent_orchestrations.items():
            logger.info(f"🔍 PROCESSOR DEBUG: Orchestration {orch_id} has {len(orch_data.get('agents', []))} agents")
            for agent in orch_data.get('agents', []):
                logger.info(f"🔍 PROCESSOR DEBUG:   Agent: {agent.get('name')} with {len(agent.get('spans', []))} spans")

        # Show details of standalone agents
        for agent_name, agent_data in self.agents.items():
            logger.info(f"🔍 PROCESSOR DEBUG: Standalone agent {agent_name} has {len(agent_data.get('spans', []))} spans")

        # Extract session information from spans
        trace_session_id = None
        chat_session_id = None
        for span in self.spans:
            if span.get('session_id'):
                trace_session_id = span['session_id']
            if span.get('chat_session_id'):
                chat_session_id = span['chat_session_id']
                break

        # For session-aware traces, we need to use the original trace_id
        # but ensure all agents are grouped under the session
        if chat_session_id:
            logger.info(f"Processing session-aware trace {trace_id} for chat session {chat_session_id}")
        else:
            logger.info(f"Processing standalone trace {trace_id}")
        
        # If we have a session, create a proper hierarchical structure
        if chat_session_id:
            # Create session orchestration agent (level 1)
            session_orchestration = {
                'name': f"chat_session_{chat_session_id[:8]}",
                'level': 1,
                'framework': 'langgraph',
                'component_type': 'session_orchestration',
                'parent_agent_id': None,
                'session_id': trace_session_id,
                'chat_session_id': chat_session_id,
                'start_time': self._get_earliest_start_time(self.spans),
                'end_time': self._get_latest_end_time(self.spans),
                'status': 'completed',
                'agents': []
            }
            
            # Group agents by orchestration and level
            agent_groups = {}
            logger.info(f"🔍 PROCESSOR DEBUG: Grouping agents from self.agents (standalone agents)")
            for agent_name, agent_data in self.agents.items():
                orchestration_id = agent_data.get('agent_orchestration_id', 'default')
                logger.info(f"🔍 PROCESSOR DEBUG: Agent {agent_name} has orchestration_id: {orchestration_id}")
                if orchestration_id not in agent_groups:
                    agent_groups[orchestration_id] = {'agents': [], 'components': []}

                if agent_data.get('level', 2) == 2:  # Agent level
                    agent_groups[orchestration_id]['agents'].append(agent_data)
                    logger.info(f"🔍 PROCESSOR DEBUG: Added {agent_name} to agent group for orchestration {orchestration_id}")
                else:  # Component level
                    agent_groups[orchestration_id]['components'].append(agent_data)
                    logger.info(f"🔍 PROCESSOR DEBUG: Added {agent_name} to component group for orchestration {orchestration_id}")

            logger.info(f"🔍 PROCESSOR DEBUG: Final agent_groups after self.agents: {list(agent_groups.keys())}")
            for orch_id, group in agent_groups.items():
                logger.info(f"🔍 PROCESSOR DEBUG: Group {orch_id}: {len(group['agents'])} agents, {len(group['components'])} components")

            # CRITICAL FIX: Also process agent orchestrations to create logical agents
            logger.info(f"🔍 PROCESSOR DEBUG: Processing agent orchestrations")
            for orch_id, orch_data in self.agent_orchestrations.items():
                if orch_id not in agent_groups:
                    agent_groups[orch_id] = {'agents': [], 'components': []}

                logger.info(f"🔍 PROCESSOR DEBUG: Processing orchestration {orch_id} with {len(orch_data.get('agents', []))} agent entries")

                # Add agents from orchestrations to agent_groups
                for agent_data in orch_data.get('agents', []):
                    agent_name = agent_data.get('name', 'unknown')
                    agent_level = agent_data.get('level', 2)

                    if agent_level == 2:  # Agent level
                        agent_groups[orch_id]['agents'].append(agent_data)
                        logger.info(f"🔍 PROCESSOR DEBUG: Added {agent_name} to agent group for orchestration {orch_id}")
                    else:  # Component level
                        agent_groups[orch_id]['components'].append(agent_data)
                        logger.info(f"🔍 PROCESSOR DEBUG: Added {agent_name} to component group for orchestration {orch_id}")

            logger.info(f"🔍 PROCESSOR DEBUG: Final agent_groups after orchestrations: {list(agent_groups.keys())}")
            for orch_id, group in agent_groups.items():
                logger.info(f"🔍 PROCESSOR DEBUG: Group {orch_id}: {len(group['agents'])} agents, {len(group['components'])} components")

            # CRITICAL FIX: Also process all spans to find LLM spans that should be aggregated
            # This is similar to how LangChain processor works
            logger.debug(f"🔍 Processing {len(self.spans)} total spans for hierarchical aggregation")
            
            # Group all spans by their parent context
            span_groups = {}
            for span in self.spans:
                # Find the parent agent for this span
                parent_agent = self._find_parent_agent_for_span(span)
                if parent_agent not in span_groups:
                    span_groups[parent_agent] = []
                span_groups[parent_agent].append(span)
            
            logger.debug(f"🔍 Grouped spans by parent: {list(span_groups.keys())}")
            for parent, spans in span_groups.items():
                logger.debug(f"🔍 Parent {parent}: {len(spans)} spans")
            
            # Process each orchestration group
            for orchestration_id, group in agent_groups.items():
                # Create agent orchestration entry (level 2)
                orchestration_entry = {
                    'name': f"agent_orchestration_{orchestration_id[:8]}" if orchestration_id != 'default' else 'agent_orchestration',
                    'level': 2,
                    'framework': 'langgraph',
                    'component_type': 'agent_orchestration',
                    'parent_agent_id': session_orchestration['name'],
                    'session_id': trace_session_id,
                    'chat_session_id': chat_session_id,
                    'start_time': self._get_earliest_start_time([span for agent in group['agents'] for span in agent['spans']]),
                    'end_time': self._get_latest_end_time([span for agent in group['agents'] for span in agent['spans']]),
                    'status': 'completed',
                    'agents': []
                }
                
                # CRITICAL FIX: Aggregate agent_data by agent_name before creating level 3 agents
                # Multiple spans with same agent_name should become one logical agent
                agents_by_name = {}
                for agent_data in group['agents']:
                    agent_name = agent_data['name']
                    if agent_name not in agents_by_name:
                        agents_by_name[agent_name] = []
                    agents_by_name[agent_name].append(agent_data)

                logger.info(f"🔍 PROCESSOR DEBUG: Orchestration {orchestration_id} has {len(agents_by_name)} unique agent names: {list(agents_by_name.keys())}")

                # Add individual agents (level 3) - now aggregated by name
                for agent_name, agent_data_list in agents_by_name.items():
                    logger.debug(f"🔍 Processing aggregated agent: {agent_name} with {len(agent_data_list)} data entries")

                    # Aggregate all spans from all agent_data entries with this name
                    all_agent_spans = []
                    for agent_data in agent_data_list:
                        all_agent_spans.extend(agent_data['spans'])
                        logger.debug(f"🔍 Added {len(agent_data['spans'])} spans from agent_data entry")

                    # Add any additional spans that belong to this agent from span groups
                    if agent_name in span_groups:
                        all_agent_spans.extend(span_groups[agent_name])
                        logger.debug(f"🔍 Added {len(span_groups[agent_name])} additional spans from span groups")

                    # Remove duplicates based on span_id or content
                    unique_spans = []
                    seen_spans = set()
                    for span in all_agent_spans:
                        span_key = span.get('span_id') or str(span.get('start_time', '')) + str(span.get('agent_name', ''))
                        if span_key not in seen_spans:
                            unique_spans.append(span)
                            seen_spans.add(span_key)

                    logger.debug(f"🔍 Total unique spans for aggregated agent {agent_name}: {len(unique_spans)}")

                    # Log span details for debugging
                    for i, span in enumerate(unique_spans):
                        logger.debug(f"🔍 Agent span {i}: {span.get('agent_name', 'unknown')} - "
                                   f"tokens: {span.get('input_tokens', 0)}+{span.get('output_tokens', 0)}, "
                                   f"cost: {span.get('cost', 0.0)}, duration: {span.get('duration', 0)}")

                    model_info = self._extract_model_info(unique_spans)
                    logger.debug(f"🔍 Extracted model info: {model_info}")

                    total_tokens = sum(span.get('input_tokens', 0) + span.get('output_tokens', 0) for span in unique_spans)
                    total_cost = sum(span.get('cost', 0) for span in unique_spans)

                    logger.debug(f"🔍 Calculated metrics for aggregated agent {agent_name} - total_tokens: {total_tokens}, total_cost: {total_cost}")

                    agent_entry = {
                        'name': agent_name,  # Use aggregated name
                        'level': 3,  # Individual agents are level 3
                        'framework': 'langgraph',
                        'component_type': 'agent',
                        'parent_agent_id': orchestration_entry['name'],
                        'session_id': trace_session_id,
                        'chat_session_id': chat_session_id,
                        'start_time': self._get_earliest_start_time(unique_spans),
                        'end_time': self._get_latest_end_time(unique_spans),
                        'duration': self._calculate_duration(unique_spans),
                        'status': 'completed',
                        'total_tokens': total_tokens,
                        'total_cost': total_cost,
                        'model_info': model_info,
                        'agents': []
                    }

                    logger.debug(f"🔍 Created aggregated agent entry: {agent_entry['name']} with {len(unique_spans)} spans, duration: {agent_entry['duration']}, tokens: {agent_entry['total_tokens']}, cost: {agent_entry['total_cost']}")

                    # Add components (level 4) as sub-agents
                    # Note: Components are not aggregated by name since they're less common
                    for component_data in group['components']:
                        if component_data.get('parent_agent_id') == agent_name:
                            # Calculate component metrics
                            component_tokens = sum(span.get('input_tokens', 0) + span.get('output_tokens', 0) for span in component_data['spans'])
                            component_cost = sum(span.get('cost', 0) for span in component_data['spans'])
                            component_model_info = self._extract_model_info(component_data['spans'])

                            logger.debug(f"🔍 Component {component_data['name']} - tokens: {component_tokens}, cost: {component_cost}")

                            component_entry = {
                                'name': component_data['name'],
                                'level': 4,  # Components are level 4
                                'framework': 'langgraph',
                                'component_type': component_data.get('component_type', 'component'),
                                'parent_agent_id': agent_entry['name'],
                                'session_id': trace_session_id,
                                'chat_session_id': chat_session_id,
                                'start_time': self._get_earliest_start_time(component_data['spans']),
                                'end_time': self._get_latest_end_time(component_data['spans']),
                                'duration': self._calculate_duration(component_data['spans']),
                                'status': 'completed',
                                'total_tokens': component_tokens,
                                'total_cost': component_cost,
                                'model_info': component_model_info,
                                'agents': []
                            }
                            agent_entry['agents'].append(component_entry)

                            # Aggregate component metrics to parent agent
                            agent_entry['total_tokens'] += component_tokens
                            agent_entry['total_cost'] += component_cost

                    orchestration_entry['agents'].append(agent_entry)
                
                session_orchestration['agents'].append(orchestration_entry)
                agents.append(orchestration_entry)
            
            # Add the session orchestration as the main agent
            agents.insert(0, session_orchestration)
        else:
            # Fallback to standalone agents if no session
            for agent_name, agent_data in self.agents.items():
                # Extract model information from spans
                model_info = self._extract_model_info(agent_data['spans'])
                
                # Calculate metrics
                total_tokens = sum(span.get('input_tokens', 0) + span.get('output_tokens', 0) for span in agent_data['spans'])
                total_cost = sum(span.get('cost', 0) for span in agent_data['spans'])
                
                agent_entry = {
                    'name': agent_name,
                    'level': 1,
                    'framework': 'langgraph',
                    'component_type': 'agent',
                    'parent_agent_id': None,
                    'session_id': trace_session_id,
                    'start_time': self._get_earliest_start_time(agent_data['spans']),
                    'end_time': self._get_latest_end_time(agent_data['spans']),
                    'duration': self._calculate_duration(agent_data['spans']),
                    'status': 'completed',
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
                    'model_info': model_info,
                    'agents': []
                }
                agents.append(agent_entry)
        
        # Create hierarchical trace
        hierarchical_trace = HierarchicalTrace(
            trace_id=trace_id,  # Use the original trace_id, not session_trace_id
            name=f"LangGraph Session" if chat_session_id else f"LangGraph Workflow",
            agents=agents,
            dependencies=dependencies,
            metadata={
                'framework': 'langgraph',
                'session_id': trace_session_id,
                'chat_session_id': chat_session_id,
                'agent_count': len(agents),
                'orchestration_count': len(self.agent_orchestrations)
            }
        )

        # 🔍 DEBUG LOGGING: Show final result
        logger.info(f"🔍 PROCESSOR DEBUG: Final hierarchical trace result:")
        logger.info(f"🔍 PROCESSOR DEBUG: Total agents in trace: {len(agents)}")
        for i, agent in enumerate(agents):
            logger.info(f"🔍 PROCESSOR DEBUG: Agent {i}: {agent.get('name')} (level {agent.get('level')}, type: {agent.get('component_type')})")
            if 'agents' in agent and agent['agents']:
                logger.info(f"🔍 PROCESSOR DEBUG:   Sub-agents: {[a.get('name') for a in agent.get('agents', [])]}")

        logger.info(f"Processed LangGraph trace {trace_id} with {len(agents)} agents")
        return hierarchical_trace
    
    def _extract_model_info(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract model information from spans."""
        model_info = {}
        
        logger.debug(f"🔍 Extracting model info from {len(spans)} spans")
        
        for i, span in enumerate(spans):
            logger.debug(f"🔍 Span {i}: {span.get('agent_name', 'unknown')} - "
                        f"model_name: {span.get('model_name')}, "
                        f"model_provider: {span.get('model_provider')}, "
                        f"input_tokens: {span.get('input_tokens', 0)}, "
                        f"output_tokens: {span.get('output_tokens', 0)}, "
                        f"cost: {span.get('cost', 0.0)}")
            
            if span.get('model_name'):
                model_info['model_name'] = span.get('model_name')
                model_info['model_provider'] = span.get('model_provider')
                model_info['model_parameters'] = span.get('model_parameters')
                logger.debug(f"🔍 Found model info: {model_info}")
                break
        
        if not model_info:
            logger.debug("🔍 No model information found in spans")
        
        return model_info
    
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
    
    def _find_parent_agent_for_span(self, span: Dict[str, Any]) -> str:
        """Find the parent agent for a span based on execution context."""
        agent_name = span.get('agent_name', '')
        component_type = span.get('component_type', '')
        
        # Check if this span has explicit parent information
        parent_span_id = span.get('parent_span_id')
        if parent_span_id:
            # Try to find parent agent from existing agents
            for existing_agent_name, agent_data in self.agents.items():
                for agent_span in agent_data['spans']:
                    if agent_span.get('span_id') == parent_span_id:
                        return existing_agent_name
        
        # Check if this is a LangGraph agent span - it should be its own parent
        if component_type == 'agent':
            # For LangGraph agents, the agent_name is already the correct name
            return agent_name
        
        # Check if this is an LLM span - it should belong to the most recent LangGraph agent
        if (agent_name.startswith('llm:') or component_type == 'llm'):
            # Find the most recently created LangGraph agent
            for existing_agent_name, agent_data in self.agents.items():
                if agent_data.get('component_type') == 'agent':
                    return existing_agent_name
            # Fallback to most recent agent
            if self.agents:
                return list(self.agents.keys())[-1]

        # Check if this is a tool span - it should belong to the most recent LangGraph agent
        if (agent_name.startswith('tool:') or component_type == 'tool'):
            # Find the most recently created LangGraph agent
            for existing_agent_name, agent_data in self.agents.items():
                if agent_data.get('component_type') == 'agent':
                    return existing_agent_name
            # Fallback to most recent agent
            if self.agents:
                return list(self.agents.keys())[-1]

        # Check if this is a chain span - it should belong to the most recent LangGraph agent
        if (agent_name.startswith('langchain:') or component_type == 'chain'):
            # Find the most recently created LangGraph agent
            for existing_agent_name, agent_data in self.agents.items():
                if agent_data.get('component_type') == 'agent':
                    return existing_agent_name
            # Fallback to most recent agent
            if self.agents:
                return list(self.agents.keys())[-1]
        
        # Default fallback - use the most recent agent
        if self.agents:
            return list(self.agents.keys())[-1]
        
        # If no agents exist yet, this span will be orphaned
        return 'orphaned'

    def detect_framework(self, span_data: Dict[str, Any]) -> bool:
        """Detect if this is a LangGraph span."""
        agent_name = span_data.get('agent_name', '')
        metadata = span_data.get('metadata', {})
        
        # Check for LangGraph indicators
        return (
            span_data.get('framework') == 'langgraph' or
            span_data.get('chat_session_id') is not None or  # LangGraph workflows have chat sessions
            # CRITICAL FIX: Also detect LLM spans that are part of LangGraph workflows
            # by checking if there's an active LangGraph trace in the same session
            (agent_name.startswith('llm:') and span_data.get('chat_session_id') is not None) or
            (span_data.get('component_type') == 'agent' and span_data.get('framework') == 'langgraph')
        )
