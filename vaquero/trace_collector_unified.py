"""Unified trace collector that routes to framework-specific processors."""

import uuid
import json
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
from .processors import LangChainProcessor
from .processors.langgraph_processor import LangGraphProcessor
from .serialization import safe_serialize
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

if TYPE_CHECKING:
    from .sdk import VaqueroSDK


def _get_trace_url(endpoint: str, trace_id: str) -> str:
    """Get the trace URL based on the endpoint configuration.
    
    Args:
        endpoint: The API endpoint URL
        trace_id: The trace ID
        
    Returns:
        The full URL to view the trace in the UI
    """
    # Check if using cloud (api.vaquero.app) or self-hosted
    # If endpoint contains api.vaquero.app, use cloud UI
    if "api.vaquero.app" in endpoint:
        return f"https://www.vaquero.app/traces/{trace_id}"
    else:
        # Self-hosted - use localhost:3000
        return f"http://localhost:3000/traces/{trace_id}"



def json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects like ChatGeneration, AIMessage, etc.
        try:
            return safe_serialize(obj)
        except Exception:
            # Fallback to string representation
            return str(obj)
    return obj

class UnifiedTraceCollector:
    """Unified trace collector that routes spans to framework-specific processors."""
    
    def __init__(self, sdk: 'VaqueroSDK'):
        self.sdk = sdk
        self.config = sdk.config
        self.processors = {
            'langchain': LangChainProcessor(),
            'langgraph': LangGraphProcessor(),
            # Add other processors as needed
        }
        self.trace_processors = {}  # trace_id -> processor
        self._traces = {}  # For compatibility with shutdown method
        self._cached_user_id = None  # Cache user ID to avoid repeated whoami calls
        self._cached_project_id = None  # Cache project ID to avoid repeated whoami calls
    
    def process_span(self, span_data: Dict[str, Any]) -> None:
        """Process span with appropriate framework processor."""
        trace_id = span_data.get('trace_id')
        if not trace_id:
            return

        # Get or create processor for this trace
        if trace_id not in self.trace_processors:
            framework = self._detect_framework(span_data)
            self.trace_processors[trace_id] = self.processors[framework]

        # Add span to appropriate processor
        processor = self.trace_processors[trace_id]
        processor.add_span(span_data)
    
    def _detect_framework(self, span_data: Dict[str, Any]) -> str:
        """Detect framework from span data."""
        # Check LangGraph first since it has more specific detection
        if self.processors['langgraph'].detect_framework(span_data):
            return 'langgraph'
        
        # Then check LangChain
        if self.processors['langchain'].detect_framework(span_data):
            return 'langchain'
        
        # Default to langchain for backward compatibility
        return 'langchain'
    
    def finalize_trace(self, trace_id: str) -> None:
        """Finalize trace and send to database."""

        if trace_id not in self.trace_processors:
            return

        try:
            processor = self.trace_processors[trace_id]
            hierarchical_trace = processor.process_trace(trace_id)
            self._send_to_database(hierarchical_trace)
        except Exception:
            pass
        finally:
            # Clean up
            if trace_id in self.trace_processors:
                del self.trace_processors[trace_id]
    
    def end_trace(self, trace_id: str, end_data: Dict[str, Any]) -> None:
        """End a trace and finalize it."""
        try:
            if trace_id in self.trace_processors:
                processor = self.trace_processors[trace_id]
                hierarchical_trace = processor.process_trace(trace_id)
                
                # Send to database
                self._send_to_database(hierarchical_trace)
                
                # Clean up
                del self.trace_processors[trace_id]
        except Exception:
            pass
    
    def _send_to_database(self, hierarchical_trace) -> None:
        """Send hierarchical trace to database."""
        # Extract trace info early for logging
        trace_id = None
        trace_name = None
        trace_url = None
        try:
            trace_id = hierarchical_trace.trace_id
            trace_name = hierarchical_trace.name or trace_id
            trace_url = _get_trace_url(self.sdk.config.endpoint, trace_id)
            # Calculate trace timing from agents
            start_times = []
            end_times = []
            for agent in hierarchical_trace.agents:
                if agent.get('start_time'):
                    # Convert string to datetime if needed
                    start_time = agent['start_time']
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    start_times.append(start_time)
                if agent.get('end_time'):
                    # Convert string to datetime if needed
                    end_time = agent['end_time']
                    if isinstance(end_time, str):
                        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    end_times.append(end_time)
            
            # Use earliest start and latest end times
            trace_start_time = min(start_times) if start_times else None
            trace_end_time = max(end_times) if end_times else None
            trace_duration_ms = 0
            if trace_start_time and trace_end_time:
                trace_duration_ms = int((trace_end_time - trace_start_time).total_seconds() * 1000)
            
            # Create trace data
            # Extract session_id from agents if available
            trace_session_id = None
            for agent in hierarchical_trace.agents:
                agent_session_id = None

                # Check agent metadata for session_id
                if agent.get('metadata') and isinstance(agent.get('metadata'), dict):
                    agent_session_id = agent['metadata'].get('session_id')

                # Check tags for session_id
                if not agent_session_id and agent.get('tags') and isinstance(agent.get('tags'), dict):
                    agent_session_id = agent['tags'].get('session_id')

                if agent_session_id:
                    trace_session_id = agent_session_id
                    break

            # Extract environment from metadata
            trace_environment = None
            if hierarchical_trace.metadata:
                trace_environment = hierarchical_trace.metadata.get('environment')
            
            # Ensure we have valid datetime values for the API
            current_time = datetime.utcnow()
            
            # Format datetimes with explicit UTC timezone indicator
            def format_datetime(dt):
                if dt:
                    return dt.isoformat() + "Z" if not dt.isoformat().endswith('Z') else dt.isoformat()
                return current_time.isoformat() + "Z"
            
            # Extract session metadata from first agent that has it
            session_metadata = None
            for agent in hierarchical_trace.agents:
                if agent.get('session_metadata'):
                    session_metadata = agent['session_metadata']
                    break

            # Extract session fields
            session_type = None
            session_timeout_minutes = None
            message_count = 0
            if session_metadata:
                session_type = session_metadata.get('session_type')
                session_timeout_minutes = session_metadata.get('timeout_minutes')
                message_count = session_metadata.get('message_count', 0)

            # Extract chat_session_id from metadata
            chat_session_id = hierarchical_trace.metadata.get('chat_session_id') if hierarchical_trace.metadata else None
            
            # Prepare raw_data with metadata for storage in database
            # The backend expects raw_data to contain metadata (as per get_trace_metadata function)
            trace_metadata = hierarchical_trace.metadata or {}
            raw_data = {
                "metadata": trace_metadata
            } if trace_metadata else None
            
            trace_data = {
                "trace_id": hierarchical_trace.trace_id,
                "name": hierarchical_trace.name,
                "status": "completed",
                "start_time": format_datetime(trace_start_time),
                "end_time": format_datetime(trace_end_time),
                "duration_ms": trace_duration_ms,
                "session_id": trace_session_id,
                "session_type": session_type,
                "session_start_time": session_metadata.get('start_time') if session_metadata else None,
                "session_end_time": session_metadata.get('end_time') if session_metadata else None,
                "session_timeout_minutes": session_timeout_minutes,
                "message_count": message_count,
                "chat_session_id": chat_session_id,
                "environment": trace_environment,
                "tags": {},
                "raw_data": raw_data
            }
            
            # Create agents data - recursively collect all agents from hierarchical structure
            agents_data = []

            def collect_agents_recursively(agent_hierarchy):
                """Recursively collect all agents from hierarchical structure."""
                # Add current agent
                agents_data.append(agent_hierarchy)
                # Recursively collect child agents
                for child in agent_hierarchy.get('agents', []):
                    collect_agents_recursively(child)

            # Collect all agents from the hierarchical structure
            for agent in hierarchical_trace.agents:
                collect_agents_recursively(agent)

            # Phase 1: Generate all agent_ids and build name->id mapping
            name_to_id = {}
            for agent in agents_data:
                agent_id = agent.get('agent_id') or str(uuid.uuid4())
                agent['agent_id'] = agent_id
                name_to_id[agent['name']] = agent_id

            # Phase 2: Resolve all parent_agent_id references from names to IDs
            for agent in agents_data:
                parent_name = agent.get('parent_agent_id')
                if parent_name and parent_name in name_to_id:
                    agent['parent_agent_id'] = name_to_id[parent_name]  # Convert name → UUID

            # Now process all collected agents
            processed_agents_data = []
            for agent in agents_data:
                # Map model fields from either flat fields or nested model_info (from LangGraph processor)
                model_info = agent.get('model_info', {}) or {}
                llm_model_name = agent.get('llm_model_name') or model_info.get('model_name')
                llm_model_provider = agent.get('llm_model_provider') or model_info.get('model_provider')
                llm_model_parameters = agent.get('llm_model_parameters') or model_info.get('model_parameters')
                system_prompt = agent.get('system_prompt')

                # Normalize token/cost fields. LangGraph processor aggregates as total_tokens/total_cost.
                input_tokens = agent.get('input_tokens') or 0
                output_tokens = agent.get('output_tokens') or 0
                total_tokens = agent.get('total_tokens') or (input_tokens + output_tokens) or agent.get('total_tokens') or agent.get('total_token_count') or 0
                if not total_tokens and isinstance(model_info, dict):
                    # Some spans only report total tokens at the model_info level (rare)
                    total_tokens = model_info.get('total_tokens', 0)
                cost_val = agent.get('cost')
                if cost_val is None:
                    cost_val = agent.get('total_cost') or 0.0

                # Format agent start_time and end_time properly
                agent_start_time = agent.get('start_time')
                agent_end_time = agent.get('end_time')
                
                if agent_start_time and isinstance(agent_start_time, str):
                    try:
                        agent_start_time = datetime.fromisoformat(agent_start_time.replace('Z', '+00:00')).isoformat() + "Z"
                    except (ValueError, AttributeError):
                        agent_start_time = None
                
                if agent_end_time and isinstance(agent_end_time, str):
                    try:
                        agent_end_time = datetime.fromisoformat(agent_end_time.replace('Z', '+00:00')).isoformat() + "Z"
                    except (ValueError, AttributeError):
                        agent_end_time = None
                
                # Extract session fields from agent metadata
                agent_session_id = None
                message_type = None
                message_content = None
                message_sequence = None
                user_message_id = None
                agent_orchestration_id = None

                # Check agent metadata for session fields
                agent_metadata = agent.get('metadata', {}) or agent.get('tags', {})
                if isinstance(agent_metadata, dict):
                    agent_session_id = agent_metadata.get('session_id') or agent_metadata.get('chat_session_id')
                    message_type = agent_metadata.get('message_type')
                    message_content = agent_metadata.get('message_content')
                    message_sequence = agent_metadata.get('message_sequence')
                    user_message_id = agent_metadata.get('user_message_id')
                    agent_orchestration_id = agent_metadata.get('agent_orchestration_id')

                # Also check direct agent fields
                if not agent_session_id:
                    agent_session_id = agent.get('session_id') or agent.get('chat_session_id')
                if not message_type:
                    message_type = agent.get('message_type')
                if not message_content:
                    message_content = agent.get('message_content')
                if not message_sequence:
                    message_sequence = agent.get('message_sequence')
                if not user_message_id:
                    user_message_id = agent.get('user_message_id')
                if not agent_orchestration_id:
                    agent_orchestration_id = agent.get('agent_orchestration_id')

                # Normalize input_data and output_data to always be dictionaries
                # API requires JSONB fields to be dicts, but spans can have lists/strings/etc.
                input_data_raw = agent.get('input_data')
                if not isinstance(input_data_raw, dict):
                    if input_data_raw is None:
                        input_data_normalized = {}
                    elif isinstance(input_data_raw, (list, tuple)):
                        input_data_normalized = {"items": input_data_raw}
                    elif isinstance(input_data_raw, str):
                        input_data_normalized = {"value": input_data_raw}
                    else:
                        input_data_normalized = {"data": input_data_raw}
                else:
                    input_data_normalized = input_data_raw

                output_data_raw = agent.get('output_data')
                if not isinstance(output_data_raw, dict):
                    if output_data_raw is None:
                        output_data_normalized = {}
                    elif isinstance(output_data_raw, (list, tuple)):
                        output_data_normalized = {"items": output_data_raw}
                    elif isinstance(output_data_raw, str):
                        output_data_normalized = {"value": output_data_raw}
                    else:
                        output_data_normalized = {"data": output_data_raw}
                else:
                    output_data_normalized = output_data_raw

                # Extract error information from agent dict
                agent_error = agent.get('error')
                error_message = None
                error_type = None
                error_stack_trace = None
                if agent_error and isinstance(agent_error, dict):
                    error_message = agent_error.get('message')
                    error_type = agent_error.get('type')
                    error_stack_trace = agent_error.get('traceback')
                # Also check for direct error fields (in case they were already extracted)
                if not error_message:
                    error_message = agent.get('error_message')
                if not error_type:
                    error_type = agent.get('error_type')
                if not error_stack_trace:
                    error_stack_trace = agent.get('error_stack_trace')

                agent_data = {
                    "trace_id": hierarchical_trace.trace_id,
                    "agent_id": agent.get('agent_id', str(uuid.uuid4())),
                    "name": agent.get('name', ''),
                    "type": agent.get('component_type', 'agent'),
                    "description": agent.get('description', ''),
                    "tags": agent.get('tags', {}),
                    "start_time": agent_start_time,
                    "end_time": agent_end_time,
                    "duration_ms": int(agent.get('duration_ms') or 0),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost_val,
                    "status": agent.get('status', 'completed'),
                    "input_data": input_data_normalized,
                    "output_data": output_data_normalized,
                    # Error fields
                    "error_message": error_message,
                    "error_type": error_type,
                    "error_stack_trace": error_stack_trace,
                    # LLM-specific fields
                    "llm_model_name": llm_model_name,
                    "llm_model_provider": llm_model_provider,
                    "llm_model_parameters": llm_model_parameters,
                    "system_prompt": system_prompt,
                    # Hierarchical fields
                    "parent_agent_id": agent.get('parent_agent_id'),
                    "level": agent.get('level', 1),
                    "framework": agent.get('framework'),
                    "component_type": agent.get('component_type'),
                    "framework_metadata": agent.get('framework_metadata', {}),
                    # Chat session fields
                    "message_type": message_type,
                    "message_content": message_content,
                    "message_sequence": message_sequence,
                    "user_message_id": user_message_id,
                    "agent_orchestration_id": agent_orchestration_id,
                    "chat_session_id": agent_session_id
                }
                processed_agents_data.append(agent_data)

            # Send as batch to API
            batch_data = {
                "traces": [trace_data],
                "agents": processed_agents_data
            }

            self._send_batch_to_api(batch_data)
            
        except Exception:
            # Log trace URL even if there's an error
            if trace_id and trace_url:
                print(f"🤠 Vaquero: Trace '{trace_name}' prepared")
                print(f"🤠 Vaquero: View trace at {trace_url}")
    
    def _send_agent_to_api(self, agent_data: dict) -> None:
        """Send individual agent to API."""
        try:
            if not REQUESTS_AVAILABLE:
                return
            
            url = f"{self.sdk.config.endpoint}/api/v1/traces"
            headers = {
                "Authorization": f"Bearer {self.sdk.config.api_key}",
                "Content-Type": "application/json"
            }
            
            requests.post(url, json=agent_data, headers=headers, timeout=30)
        except Exception:
            pass
    
    def _send_batch_to_api(self, batch_data: dict) -> None:
        """Send batch trace data to API."""
        # Extract trace info for logging before the try block
        traces_to_log = []
        if batch_data.get("traces"):
            for trace in batch_data["traces"]:
                trace_id = trace.get("trace_id")
                if trace_id:
                    trace_url = _get_trace_url(self.sdk.config.endpoint, trace_id)
                    trace_name = trace.get("name") or trace.get("trace_id", "trace")
                    traces_to_log.append((trace_id, trace_name, trace_url))
        
        try:
            if not REQUESTS_AVAILABLE:
                # Still log the trace URL even if requests isn't available
                for trace_id, trace_name, trace_url in traces_to_log:
                    print(f"🤠 Vaquero: Trace '{trace_name}' prepared (requests library not available)")
                    print(f"🤠 Vaquero: View trace at {trace_url}")
                return
            
            serializable_data = json_serializable(batch_data)
            user_id = self._get_user_id()
            project_id = self._cached_project_id
            
            url = f"{self.sdk.config.endpoint}/api/v1/traces/batch"
            headers = {
                "Authorization": f"Bearer {self.sdk.config.api_key}",
                "Content-Type": "application/json"
            }
            
            if user_id:
                headers["X-User-ID"] = user_id
            
            if project_id:
                headers["X-Project-ID"] = project_id

            response = requests.post(url, json=serializable_data, headers=headers, timeout=30)
            
            # Log trace links for successfully sent traces
            # 200 = OK, 201 = Created, 202 = Accepted (all are success codes)
            if response.status_code in (200, 201, 202):
                for trace_id, trace_name, trace_url in traces_to_log:
                    print(f"🤠 Vaquero: Trace '{trace_name}' sent successfully")
                    print(f"🤠 Vaquero: View trace at {trace_url}")
            else:
                # Log even on error, but indicate it may have failed
                for trace_id, trace_name, trace_url in traces_to_log:
                    print(f"🤠 Vaquero: Trace '{trace_name}' sent (status: {response.status_code})")
                    print(f"🤠 Vaquero: View trace at {trace_url}")
        except Exception as e:
            # Log trace URLs even if there's an error
            for trace_id, trace_name, trace_url in traces_to_log:
                print(f"🤠 Vaquero: Trace '{trace_name}' prepared")
                print(f"🤠 Vaquero: View trace at {trace_url}")
    
    def _get_user_id(self) -> Optional[str]:
        """Get user ID and project ID from whoami endpoint (cached)."""
        # Return cached user ID if available
        if self._cached_user_id is not None:
            return self._cached_user_id
            
        try:
            if not REQUESTS_AVAILABLE:
                return None
            
            whoami_url = f"{self.sdk.config.endpoint}/api/v1/auth/whoami"
            whoami_headers = {
                "Authorization": f"Bearer {self.sdk.config.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(whoami_url, headers=whoami_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                user_id = data.get("user_id")
                project_id = self.sdk.config.project_id
                
                self._cached_user_id = user_id
                self._cached_project_id = project_id
                
                return user_id
            return None
        except Exception:
            return None
    
    def shutdown(self):
        """Shutdown the unified trace collector and finalize all pending traces."""
        pending_traces = list(self.trace_processors.keys())
        
        for trace_id in pending_traces:
            try:
                self.finalize_trace(trace_id)
            except Exception:
                pass
