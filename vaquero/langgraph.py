"""
LangGraph integration for Vaquero SDK.

This module provides specialized integration for LangGraph applications,
handling the unique timing and execution flow of LangGraph agent swarms.
"""

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid
import asyncio
from contextlib import asynccontextmanager

from .langchain import VaqueroCallbackHandler
from .trace_collector_unified import UnifiedTraceCollector

logger = logging.getLogger(__name__)


class VaqueroLangGraphHandler(VaqueroCallbackHandler):
    """
    Specialized Vaquero handler for LangGraph applications.
    
    This handler addresses the timing issues specific to LangGraph's
    agent swarm architecture by ensuring proper trace finalization
    and sending before SDK shutdown.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_traces: Dict[str, Dict[str, Any]] = {}
        self._trace_finalization_pending = False
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain start for LangGraph."""
        try:
            super().on_chain_start(
                serialized, inputs, run_id=run_id, parent_run_id=parent_run_id,
                tags=tags, metadata=metadata, **kwargs
            )
            
            # Track active traces for LangGraph
            if run_id not in self._active_traces:
                self._active_traces[run_id] = {
                    'start_time': datetime.utcnow(),
                    'parent_run_id': parent_run_id,
                    'chain_type': serialized.get('name', 'unknown') if serialized else 'unknown',
                    'inputs': inputs
                }
        except Exception as e:
            logger.warning(f"Error in LangGraph chain start callback: {e}")
            # Still track the trace even if parent callback fails
            if run_id not in self._active_traces:
                self._active_traces[run_id] = {
                    'start_time': datetime.utcnow(),
                    'parent_run_id': parent_run_id,
                    'chain_type': 'unknown',
                    'inputs': inputs
                }
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain end for LangGraph."""
        try:
            super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        except Exception as e:
            logger.warning(f"Error in LangGraph chain end callback: {e}")
        
        # Update trace info
        if run_id in self._active_traces:
            self._active_traces[run_id]['end_time'] = datetime.utcnow()
            self._active_traces[run_id]['outputs'] = outputs
    
    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain error for LangGraph."""
        try:
            super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        except Exception as e:
            logger.warning(f"Error in LangGraph chain error callback: {e}")
        
        # Mark trace as errored
        if run_id in self._active_traces:
            self._active_traces[run_id]['error'] = str(error)
            self._active_traces[run_id]['end_time'] = datetime.utcnow()
    
    def finalize_langgraph_traces(self):
        """Finalize all LangGraph traces and ensure they are sent."""
        logger.debug("Finalizing LangGraph traces...")
        
        # Get all active traces
        active_trace_ids = list(self._active_traces.keys())
        
        for run_id in active_trace_ids:
            trace_info = self._active_traces[run_id]
            
            # Calculate duration
            if 'end_time' in trace_info:
                duration = (trace_info['end_time'] - trace_info['start_time']).total_seconds()
            else:
                duration = 0
            
            logger.debug(f"Finalizing LangGraph trace {run_id}: {duration:.3f}s")
            
            # Create trace data with proper datetime formatting
            start_time = trace_info['start_time']
            end_time = trace_info.get('end_time', datetime.utcnow())
            
            trace_data = {
                'run_id': run_id,
                'parent_run_id': trace_info.get('parent_run_id'),
                'chain_type': trace_info.get('chain_type', 'unknown'),
                'duration': duration,
                'inputs': trace_info.get('inputs', {}),
                'outputs': trace_info.get('outputs', {}),
                'error': trace_info.get('error'),
                'start_time': start_time.isoformat() + 'Z' if start_time else datetime.utcnow().isoformat() + 'Z',
                'end_time': end_time.isoformat() + 'Z' if end_time else datetime.utcnow().isoformat() + 'Z'
            }
            
            # Send trace data to collector
            self._send_trace_data(trace_data)
        
        # Clear active traces
        self._active_traces.clear()
        self._trace_finalization_pending = False
        
        logger.debug("LangGraph traces finalized and sent")
    
    def _send_trace_data(self, trace_data: Dict[str, Any]):
        """Send trace data to the unified trace collector."""
        try:
            # Get the trace collector
            if hasattr(self, 'trace_collector') and self.trace_collector:
                # Create a trace entry
                trace_entry = {
                    'trace_id': str(uuid.uuid4()),
                    'agent_name': f"langgraph:{trace_data['chain_type']}",
                    'function_name': trace_data['chain_type'],
                    'start_time': trace_data['start_time'],
                    'end_time': trace_data['end_time'],
                    'status': 'error' if trace_data.get('error') else 'success',
                    'duration': trace_data['duration'],
                    'inputs': trace_data['inputs'],
                    'outputs': trace_data['outputs'],
                    'error': trace_data.get('error')
                }
                
                # Send to trace collector
                self.trace_collector.add_trace(trace_entry)
                logger.debug(f"Sent LangGraph trace data: {trace_entry['trace_id']}")
            else:
                logger.warning("No trace collector available for LangGraph traces")
                
        except Exception as e:
            logger.error(f"Failed to send LangGraph trace data: {e}")


@asynccontextmanager
async def langgraph_trace_context(handler: VaqueroLangGraphHandler):
    """
    Context manager for LangGraph tracing that ensures proper cleanup.
    
    This context manager ensures that traces are finalized and sent
    before the context exits, addressing the timing issues with
    LangGraph's execution flow.
    """
    try:
        yield handler
    finally:
        # Ensure traces are finalized before context exit
        handler.finalize_langgraph_traces()
        
        # Give a moment for traces to be sent
        await asyncio.sleep(0.1)


def get_vaquero_langgraph_handler(**kwargs) -> VaqueroLangGraphHandler:
    """
    Get a Vaquero handler specifically configured for LangGraph.
    
    Args:
        **kwargs: Additional arguments for the handler
        
    Returns:
        VaqueroLangGraphHandler: Configured handler for LangGraph
    """
    return VaqueroLangGraphHandler(**kwargs)


def create_langgraph_config(handler: VaqueroLangGraphHandler) -> Dict[str, Any]:
    """
    Create a configuration dict for LangGraph with the Vaquero handler.
    
    Args:
        handler: VaqueroLangGraphHandler instance
        
    Returns:
        Dict containing the configuration for LangGraph
    """
    return {
        'callbacks': [handler],
        'configurable': {
            'vaquero_handler': handler
        }
    }
