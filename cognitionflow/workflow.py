"""Workflow orchestration for CognitionFlow SDK.

This module provides high-level workflow APIs that automatically handle
span hierarchy, dependency tracking, and graph construction.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum

from .context import get_current_span, span_context
from .models import TraceData, SpanStatus


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""
    name: str
    function: Callable
    depends_on: List[str] = field(default_factory=list)
    agent_type: str = "generic"
    step_type: WorkflowStepType = WorkflowStepType.SEQUENTIAL
    parallel_group: Optional[str] = None
    condition: Optional[Callable] = None
    description: Optional[str] = None
    result: Any = None
    span_id: Optional[str] = None
    executed: bool = False
    error: Optional[Exception] = None


class Workflow:
    """High-level workflow orchestration with automatic span hierarchy.
    
    This class provides a declarative way to define workflows with automatic
    dependency resolution, span hierarchy creation, and graph construction.
    
    Example:
        workflow = cognitionflow.workflow("data_processing")
        workflow.add_step("clean", clean_data)
        workflow.add_step("analyze", analyze_data, depends_on=["clean"])
        workflow.add_step("report", generate_report, depends_on=["analyze"])
        
        result = await workflow.execute(input_data)
    """
    
    def __init__(self, name: str):
        """Initialize a new workflow.
        
        Args:
            name: Name of the workflow for identification
        """
        self.name = name
        self.steps: Dict[str, WorkflowStep] = {}
        self.parallel_groups: Dict[str, List[str]] = {}
        self.execution_order: List[str] = []
        self.results: Dict[str, Any] = {}
        self.workflow_span: Optional[TraceData] = None
        
    def add_step(
        self,
        name: str,
        function: Callable,
        depends_on: Optional[List[str]] = None,
        agent_type: str = "generic",
        step_type: WorkflowStepType = WorkflowStepType.SEQUENTIAL,
        parallel_group: Optional[str] = None,
        condition: Optional[Callable] = None,
        description: Optional[str] = None
    ) -> None:
        """Add a step to the workflow.
        
        Args:
            name: Unique name for the step
            function: Function to execute for this step
            depends_on: List of step names this step depends on
            agent_type: Type of agent (generic, llm, tool, etc.)
            step_type: Type of step execution
            parallel_group: Group name for parallel execution
            condition: Optional condition function for conditional steps
            description: Human-readable description of what this step does
        """
        if name in self.steps:
            raise ValueError(f"Step '{name}' already exists in workflow")
        
        # Validate dependencies
        if depends_on:
            for dep in depends_on:
                if dep not in self.steps:
                    raise ValueError(f"Dependency '{dep}' not found in workflow")
        
        step = WorkflowStep(
            name=name,
            function=function,
            depends_on=depends_on or [],
            agent_type=agent_type,
            step_type=step_type,
            parallel_group=parallel_group,
            condition=condition,
            description=description
        )
        
        self.steps[name] = step
        
        # Track parallel groups
        if parallel_group:
            if parallel_group not in self.parallel_groups:
                self.parallel_groups[parallel_group] = []
            self.parallel_groups[parallel_group].append(name)
    
    def add_parallel_steps(
        self,
        steps: List[tuple],
        group_name: str,
        depends_on: Optional[List[str]] = None
    ) -> None:
        """Add multiple steps that should execute in parallel.
        
        Args:
            steps: List of (name, function, agent_type) or (name, function, agent_type, description) tuples
            group_name: Name for the parallel group
            depends_on: Dependencies for the entire parallel group
        """
        for step_data in steps:
            if len(step_data) == 3:
                name, function, agent_type = step_data
                description = None
            elif len(step_data) == 4:
                name, function, agent_type, description = step_data
            else:
                raise ValueError("Steps must be (name, function, agent_type) or (name, function, agent_type, description) tuples")
            
            self.add_step(
                name=name,
                function=function,
                depends_on=depends_on,
                agent_type=agent_type,
                step_type=WorkflowStepType.PARALLEL,
                parallel_group=group_name,
                description=description
            )
    
    def _calculate_execution_order(self) -> List[str]:
        """Calculate the execution order based on dependencies."""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_name: str):
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{step_name}'")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            step = self.steps[step_name]
            
            # Visit dependencies first
            for dep in step.depends_on:
                visit(dep)
            
            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)
        
        return order
    
    async def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the workflow with automatic span hierarchy.
        
        Args:
            input_data: Initial input data for the workflow
            
        Returns:
            Dictionary containing results from all steps
            
        Raises:
            ValueError: If workflow has circular dependencies
            RuntimeError: If step execution fails
        """
        # Calculate execution order
        self.execution_order = self._calculate_execution_order()
        
        # Create main workflow span
        with self._create_workflow_span() as workflow_span:
            self.workflow_span = workflow_span
            workflow_span.set_attribute("workflow_name", self.name)
            workflow_span.set_attribute("total_steps", len(self.steps))
            workflow_span.set_attribute("input_type", type(input_data).__name__)
            print(f"    🔍 Created workflow span: {workflow_span.span_id}")
            
            # Set the workflow span as the current span context
            from .context import span_context
            with span_context(workflow_span):
                try:
                    # Execute steps in dependency order
                    await self._execute_steps(input_data)
                    
                    workflow_span.set_attribute("status", "completed")
                    workflow_span.set_attribute("results_count", len(self.results))
                    print(f"    🔍 Workflow span completed: {workflow_span.span_id}")
                    
                    return {
                        "workflow_id": workflow_span.trace_id,
                        "workflow_name": self.name,
                        "status": "completed",
                        "results": self.results,
                        "execution_order": self.execution_order,
                        "total_steps": len(self.steps)
                    }
                    
                except Exception as e:
                    workflow_span.set_attribute("status", "failed")
                    workflow_span.set_error(e)
                    raise
    
    def _create_workflow_span(self):
        """Create the main workflow span."""
        from .sdk import get_current_sdk
        
        sdk = get_current_sdk()
        if not sdk:
            raise RuntimeError("No SDK instance found. Call cognitionflow.initialize() first.")
        
        return sdk.span(f"workflow_{self.name}")
    
    async def _execute_steps(self, input_data: Any) -> None:
        """Execute workflow steps in dependency order."""
        # Group steps by execution phase
        execution_phases = self._group_steps_by_phase()
        
        for phase, step_names in execution_phases.items():
            if len(step_names) == 1:
                # Sequential execution
                await self._execute_step(step_names[0], input_data)
            else:
                # Parallel execution
                await self._execute_parallel_steps(step_names, input_data)
    
    def _group_steps_by_phase(self) -> Dict[int, List[str]]:
        """Group steps by execution phase based on dependencies."""
        phases = {}
        step_phases = {}
        
        for step_name in self.execution_order:
            step = self.steps[step_name]
            
            if not step.depends_on:
                # No dependencies - phase 0
                phase = 0
            else:
                # Phase is max dependency phase + 1
                phase = max(step_phases[dep] for dep in step.depends_on) + 1
            
            step_phases[step_name] = phase
            
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(step_name)
        
        return phases
    
    async def _execute_step(self, step_name: str, input_data: Any) -> None:
        """Execute a single workflow step."""
        step = self.steps[step_name]
        
        # Check condition if present
        if step.condition and not step.condition(input_data):
            return
        
        # Gather inputs from dependencies
        step_inputs = self._gather_step_inputs(step_name, input_data)
        
        # Create child span for this step
        with self._create_step_span(step, step_inputs) as step_span:
            step.span_id = step_span.span_id
            step_span.set_attribute("step_name", step_name)
            step_span.set_attribute("agent_type", step.agent_type)
            step_span.set_attribute("step_type", step.step_type.value)
            if step.description:
                step_span.set_attribute("description", step.description)
            print(f"      🔍 Created step span: {step_span.span_id} for {step_name}")

            try:
                # Set input data on the span
                step_span.inputs = {"step_inputs": step_inputs}
                
                # Execute the step function
                if asyncio.iscoroutinefunction(step.function):
                    result = await step.function(step_inputs)
                else:
                    result = step.function(step_inputs)
                
                # Set output data on the span
                step_span.outputs = {"result": result}
                
                # Apply token counting to both input data and result
                from .sdk import get_current_sdk
                sdk = get_current_sdk()
                if sdk and sdk.config.capture_tokens:
                    try:
                        from .token_counter import count_tokens, extract_tokens_from_llm_response
                        
                        # Count input tokens from step inputs
                        input_tokens = 0
                        if step_inputs:
                            input_text = str(step_inputs)
                            input_tokens = count_tokens(input_text)
                        
                        # Count output tokens from result (for LLM calls)
                        output_tokens = 0
                        total_tokens = input_tokens
                        cost = 0.0
                        model_name = None
                        provider = None
                        
                        if result:
                            # Try to extract tokens from LLM response
                            token_count = extract_tokens_from_llm_response(result)
                            if token_count.total_tokens > 0:
                                # This is an LLM response
                                input_tokens = token_count.input_tokens
                                output_tokens = token_count.output_tokens
                                total_tokens = token_count.total_tokens
                                cost = token_count.cost
                                model_name = token_count.model
                                provider = token_count.provider
                            else:
                                # This is not an LLM response, count output text tokens
                                output_text = str(result)
                                output_tokens = count_tokens(output_text)
                                total_tokens = input_tokens + output_tokens
                        
                        # Set token information on the span
                        step_span.input_tokens = input_tokens
                        step_span.output_tokens = output_tokens
                        step_span.total_tokens = total_tokens
                        step_span.cost = cost
                        step_span.model_name = model_name
                        step_span.model_provider = provider
                        
                        if total_tokens > 0:
                            print(f"      🔍 Token count: {total_tokens} tokens (input: {input_tokens}, output: {output_tokens})")
                        
                    except Exception as e:
                        print(f"      ⚠️ Token counting failed: {e}")
                
                step.result = result
                step.executed = True
                self.results[step_name] = result
                
                step_span.set_attribute("status", "completed")
                step_span.set_attribute("result_type", type(result).__name__)
                print(f"      🔍 Step span completed: {step_span.span_id}")
                
            except Exception as e:
                step.error = e
                step_span.set_attribute("status", "failed")
                step_span.set_error(e)
                raise
    
    async def _execute_parallel_steps(self, step_names: List[str], input_data: Any) -> None:
        """Execute multiple steps in parallel."""
        # Create tasks for parallel execution
        tasks = []
        for step_name in step_names:
            task = asyncio.create_task(self._execute_step(step_name, input_data))
            tasks.append((step_name, task))
        
        # Wait for all tasks to complete
        for step_name, task in tasks:
            try:
                await task
            except Exception as e:
                # Cancel remaining tasks
                for _, remaining_task in tasks:
                    if not remaining_task.done():
                        remaining_task.cancel()
                raise
    
    def _create_step_span(self, step: WorkflowStep, step_inputs: Any):
        """Create a child span for a workflow step."""
        from .sdk import get_current_sdk
        from .context import span_context
        
        sdk = get_current_sdk()
        
        # Create child span using the current span context (workflow span)
        return sdk.span(
            f"step_{step.name}",
            tags={
                "workflow_step": step.name,
                "agent_type": step.agent_type,
                "step_type": step.step_type.value,
                "depends_on": ",".join(step.depends_on) if step.depends_on else ""
            },
            metadata={
                "step_inputs": step_inputs,
                "parallel_group": step.parallel_group,
                "dependencies": step.depends_on,
                "description": step.description
            }
        )
    
    def _gather_step_inputs(self, step_name: str, input_data: Any) -> Any:
        """Gather inputs for a step from its dependencies."""
        step = self.steps[step_name]
        
        # If step has dependencies, pass dependency results
        if step.depends_on:
            # For steps with dependencies, pass the results from dependencies
            if len(step.depends_on) == 1:
                # Single dependency - pass result directly
                dep_name = step.depends_on[0]
                if dep_name in self.results:
                    return self.results[dep_name]
            else:
                # Multiple dependencies - pass as dictionary
                inputs = {}
                for dep_name in step.depends_on:
                    if dep_name in self.results:
                        inputs[dep_name] = self.results[dep_name]
                return inputs
        
        # If no dependencies, pass input_data directly
        return input_data
    
    def get_execution_plan(self) -> Dict[str, Any]:
        """Get the execution plan for this workflow.
        
        Returns:
            Dictionary containing workflow structure and execution order
        """
        return {
            "workflow_name": self.name,
            "total_steps": len(self.steps),
            "execution_order": self._calculate_execution_order(),
            "parallel_groups": self.parallel_groups,
            "dependencies": {
                name: step.depends_on 
                for name, step in self.steps.items()
            }
        }


def workflow(name: str) -> Workflow:
    """Create a new workflow.
    
    Args:
        name: Name of the workflow
        
    Returns:
        New Workflow instance
    """
    return Workflow(name)
