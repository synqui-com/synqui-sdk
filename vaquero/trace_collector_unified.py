"""Unified trace collector that routes to framework-specific processors."""

import logging
import uuid
from typing import Dict, Any, Optional, TYPE_CHECKING
from .processors import LangChainProcessor

if TYPE_CHECKING:
    from .sdk import VaqueroSDK

logger = logging.getLogger(__name__)

class UnifiedTraceCollector:
    """Unified trace collector that routes spans to framework-specific processors."""
    
    def __init__(self, sdk: 'VaqueroSDK'):
        self.sdk = sdk
        self.config = sdk.config
        self.processors = {
            'langchain': LangChainProcessor(),
            # Add other processors as needed
        }
        self.trace_processors = {}  # trace_id -> processor
        self._traces = {}  # For compatibility with shutdown method
    
    def process_span(self, span_data: Dict[str, Any]) -> None:
        """Process span with appropriate framework processor."""
        trace_id = span_data.get('trace_id')
        if not trace_id:
            logger.warning("Span has no trace_id, skipping")
            return
        
        # Get or create processor for this trace
        if trace_id not in self.trace_processors:
            framework = self._detect_framework(span_data)
            self.trace_processors[trace_id] = self.processors[framework]
            logger.debug(f"Created {framework} processor for trace {trace_id}")
        
        # Add span to appropriate processor
        processor = self.trace_processors[trace_id]
        processor.add_span(span_data)
    
    def _detect_framework(self, span_data: Dict[str, Any]) -> str:
        """Detect framework from span data."""
        for framework, processor in self.processors.items():
            if processor.detect_framework(span_data):
                return framework
        
        # Default to langchain for now
        return 'langchain'
    
    def finalize_trace(self, trace_id: str) -> None:
        """Finalize trace and send to database."""
        if trace_id not in self.trace_processors:
            logger.warning(f"No processor found for trace {trace_id}")
            return
        
        try:
            processor = self.trace_processors[trace_id]
            hierarchical_trace = processor.process_trace(trace_id)
            
            # Send to database
            self._send_to_database(hierarchical_trace)
            
            logger.info(f"Finalized trace {trace_id} with {len(hierarchical_trace.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to finalize trace {trace_id}: {e}")
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
                
                logger.info(f"Finalized trace {trace_id} with {len(hierarchical_trace.agents)} agents")
                
                # Clean up
                del self.trace_processors[trace_id]
            else:
                logger.warning(f"No processor found for trace {trace_id}")
                
        except Exception as e:
            logger.error(f"Failed to end trace {trace_id}: {e}")
    
    def _send_to_database(self, hierarchical_trace) -> None:
        """Send hierarchical trace to database."""
        try:
            # Calculate trace timing from agents
            start_times = []
            end_times = []
            for agent in hierarchical_trace.agents:
                if agent.get('start_time'):
                    # Convert string to datetime if needed
                    start_time = agent['start_time']
                    if isinstance(start_time, str):
                        from datetime import datetime
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    start_times.append(start_time)
                if agent.get('end_time'):
                    # Convert string to datetime if needed
                    end_time = agent['end_time']
                    if isinstance(end_time, str):
                        from datetime import datetime
                        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    end_times.append(end_time)
            
            # Use earliest start and latest end times
            trace_start_time = min(start_times) if start_times else None
            trace_end_time = max(end_times) if end_times else None
            trace_duration_ms = 0
            if trace_start_time and trace_end_time:
                trace_duration_ms = int((trace_end_time - trace_start_time).total_seconds() * 1000)
            
            # Create trace data
            trace_data = {
                "trace_id": hierarchical_trace.trace_id,
                "name": hierarchical_trace.name,
                "status": "completed",
                "start_time": trace_start_time.isoformat() if trace_start_time else None,
                "end_time": trace_end_time.isoformat() if trace_end_time else None,
                "duration_ms": trace_duration_ms,
                "session_id": None,  # Will be extracted from agents if available
                "tags": {},
                "metadata": {}
            }
            
            # Create agents data
            agents_data = []
            for agent in hierarchical_trace.agents:
                agent_data = {
                    "trace_id": hierarchical_trace.trace_id,
                    "agent_id": agent.get('agent_id', str(uuid.uuid4())),
                    "name": agent.get('name', ''),
                    "type": agent.get('component_type', 'agent'),
                    "description": agent.get('description', ''),
                    "tags": agent.get('tags', {}),
                    "start_time": agent.get('start_time'),
                    "end_time": agent.get('end_time'),
                    "duration_ms": int(agent.get('duration_ms', 0)),
                    "input_tokens": agent.get('input_tokens', 0),
                    "output_tokens": agent.get('output_tokens', 0),
                    "total_tokens": agent.get('total_tokens', 0),
                    "cost": agent.get('cost', 0.0),
                    "status": agent.get('status', 'completed'),
                    "input_data": agent.get('input_data', {}),
                    "output_data": agent.get('output_data', {}),
                    # Hierarchical fields
                    "parent_agent_id": agent.get('parent_agent_id'),
                    "level": agent.get('level', 1),
                    "framework": agent.get('framework'),
                    "component_type": agent.get('component_type'),
                    "framework_metadata": agent.get('framework_metadata', {})
                }
                agents_data.append(agent_data)
            
            # Send as batch to API
            batch_data = {
                "traces": [trace_data],
                "agents": agents_data
            }
            
            self._send_batch_to_api(batch_data)
            
            logger.info(f"Sent hierarchical trace to database: {hierarchical_trace.trace_id}")
            logger.info(f"  Logical agents: {len([a for a in hierarchical_trace.agents if a.get('level') == 1])}")
            logger.info(f"  Internal components: {len([a for a in hierarchical_trace.agents if a.get('level') == 2])}")
            
        except Exception as e:
            logger.error(f"Failed to send hierarchical trace to database: {e}", exc_info=True)
    
    def _send_agent_to_api(self, agent_data: dict) -> None:
        """Send individual agent to API."""
        try:
            import requests
            
            # Send to the traces API endpoint
            url = f"{self.sdk.config.endpoint}/api/v1/traces"
            headers = {
                "Authorization": f"Bearer {self.sdk.config.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=agent_data, headers=headers, timeout=30)
            
            if response.status_code not in [200, 201]:
                logger.warning(f"Failed to send agent to API: {response.status_code} - {response.text}")
            else:
                logger.debug(f"Successfully sent agent {agent_data.get('name')} to API")
                
        except Exception as e:
            logger.error(f"Failed to send agent to API: {e}")
    
    def _send_batch_to_api(self, batch_data: dict) -> None:
        """Send batch trace data to API."""
        try:
            import requests
            
            # Send to the batch traces API endpoint
            url = f"{self.sdk.config.endpoint}/api/v1/traces/batch"
            headers = {
                "Authorization": f"Bearer {self.sdk.config.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=batch_data, headers=headers, timeout=30)
            
            if response.status_code not in [200, 201, 202]:
                logger.warning(f"Failed to send batch to API: {response.status_code} - {response.text}")
            else:
                logger.debug(f"Successfully sent batch to API")
                
        except Exception as e:
            logger.error(f"Failed to send batch to API: {e}")
    
    def shutdown(self):
        """Shutdown the unified trace collector."""
        logger.debug("Unified trace collector shutdown")
