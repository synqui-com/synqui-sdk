#!/usr/bin/env python3
"""
Basic Usage Examples for CognitionFlow SDK

This file demonstrates the core functionality of the CognitionFlow SDK
with practical examples for different use cases.
"""

import asyncio
import time
import json
from typing import Dict, List, Any
import cognitionflow

# Configure the SDK
cognitionflow.configure(
    api_key="your-api-key-here",
    project_id="your-project-id",
    endpoint="https://api.cognitionflow.com",
    environment="development",
    debug=True
)

# Example 1: Simple synchronous function tracing
@cognitionflow.trace(agent_name="data_processor")
def process_user_data(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Process user data with automatic tracing."""
    # Simulate some processing time
    time.sleep(0.1)
    
    result = {
        "user_id": user_id,
        "processed_at": time.time(),
        "data_size": len(str(data)),
        "status": "processed"
    }
    
    return result

# Example 2: Asynchronous function tracing
@cognitionflow.trace(agent_name="api_client")
async def fetch_user_profile(user_id: str) -> Dict[str, Any]:
    """Fetch user profile from external API."""
    # Simulate API call delay
    await asyncio.sleep(0.2)
    
    return {
        "user_id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "created_at": "2024-01-01T00:00:00Z"
    }

# Example 3: Function with error handling
@cognitionflow.trace(agent_name="risky_operation")
def risky_operation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Operation that might fail - demonstrates error tracing."""
    if "error" in data:
        raise ValueError(f"Simulated error for data: {data}")
    
    return {"status": "success", "processed_data": data}

# Example 4: Manual span creation with context
async def complex_workflow():
    """Example of manual span creation for complex workflows."""
    async with cognitionflow.span("complex_workflow") as span:
        span.set_attribute("workflow_type", "user_onboarding")
        span.set_attribute("version", "1.0")
        
        # Step 1: Fetch user data
        async with cognitionflow.span("fetch_user_data") as fetch_span:
            fetch_span.set_attribute("source", "database")
            user_data = await fetch_user_profile("123")
        
        # Step 2: Process the data
        async with cognitionflow.span("process_data") as process_span:
            process_span.set_attribute("algorithm", "standard_processing")
            processed_data = process_user_data("123", user_data)
        
        # Step 3: Validate results
        async with cognitionflow.span("validate_results") as validate_span:
            validate_span.set_attribute("validation_type", "completeness")
            if not processed_data.get("status") == "processed":
                raise ValueError("Data processing failed validation")
        
        return processed_data

# Example 5: Batch processing with tracing
@cognitionflow.trace(agent_name="batch_processor")
def process_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of items with tracing."""
    results = []
    
    for i, item in enumerate(items):
        # Create a nested span for each item
        with cognitionflow.span(f"process_item_{i}") as item_span:
            item_span.set_attribute("item_index", i)
            item_span.set_attribute("item_type", item.get("type", "unknown"))
            
            # Simulate processing
            time.sleep(0.05)
            
            result = {
                "item_id": item.get("id", i),
                "processed": True,
                "timestamp": time.time()
            }
            results.append(result)
    
    return results

# Example 6: Custom tags and metadata
@cognitionflow.trace(
    agent_name="ml_predictor",
    tags={"model_version": "v2.1", "environment": "production"},
    metadata={"team": "ml", "cost_center": "engineering"}
)
def ml_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """ML prediction with custom tags and metadata."""
    # Simulate ML processing
    time.sleep(0.3)
    
    return {
        "prediction": 0.85,
        "confidence": 0.92,
        "model": "v2.1",
        "input_features": len(input_data)
    }

# Example 7: Conditional tracing
def conditional_operation(data: Dict[str, Any], enable_tracing: bool = True):
    """Operation that can be traced conditionally."""
    if enable_tracing:
        @cognitionflow.trace(agent_name="conditional_processor")
        def traced_operation():
            return process_user_data("conditional", data)
        return traced_operation()
    else:
        return process_user_data("conditional", data)

# Example 8: Error recovery with tracing
@cognitionflow.trace(agent_name="resilient_processor")
def resilient_operation(data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Operation with retry logic and error tracing."""
    for attempt in range(max_retries):
        try:
            with cognitionflow.span(f"attempt_{attempt + 1}") as attempt_span:
                attempt_span.set_attribute("attempt_number", attempt + 1)
                attempt_span.set_attribute("max_retries", max_retries)
                
                # Simulate operation that might fail
                if attempt < 2 and "fail" in data:
                    raise ConnectionError(f"Simulated connection error on attempt {attempt + 1}")
                
                return {"status": "success", "attempt": attempt + 1, "data": data}
                
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                raise
            time.sleep(0.1 * (attempt + 1))  # Exponential backoff

def main():
    """Run all examples."""
    print("🚀 Running CognitionFlow SDK Examples")
    print("=" * 50)
    
    # Example 1: Simple sync function
    print("\n1. Simple synchronous function tracing:")
    result1 = process_user_data("user123", {"name": "John", "age": 30})
    print(f"   Result: {result1}")
    
    # Example 2: Async function
    print("\n2. Asynchronous function tracing:")
    async def run_async_example():
        result2 = await fetch_user_profile("user456")
        print(f"   Result: {result2}")
        return result2
    
    asyncio.run(run_async_example())
    
    # Example 3: Error handling
    print("\n3. Error handling and tracing:")
    try:
        risky_operation({"error": True})
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    # Example 4: Complex workflow
    print("\n4. Complex workflow with manual spans:")
    async def run_complex_workflow():
        result4 = await complex_workflow()
        print(f"   Result: {result4}")
        return result4
    
    asyncio.run(run_complex_workflow())
    
    # Example 5: Batch processing
    print("\n5. Batch processing:")
    batch_data = [
        {"id": 1, "type": "user", "data": "user1"},
        {"id": 2, "type": "order", "data": "order1"},
        {"id": 3, "type": "product", "data": "product1"}
    ]
    result5 = process_batch(batch_data)
    print(f"   Processed {len(result5)} items")
    
    # Example 6: ML prediction with custom tags
    print("\n6. ML prediction with custom tags:")
    ml_input = {"feature1": 0.5, "feature2": 0.8, "feature3": 0.3}
    result6 = ml_prediction(ml_input)
    print(f"   Prediction: {result6}")
    
    # Example 7: Conditional tracing
    print("\n7. Conditional tracing:")
    result7a = conditional_operation({"test": "data"}, enable_tracing=True)
    result7b = conditional_operation({"test": "data"}, enable_tracing=False)
    print(f"   With tracing: {result7a}")
    print(f"   Without tracing: {result7b}")
    
    # Example 8: Resilient operation
    print("\n8. Resilient operation with retries:")
    try:
        result8 = resilient_operation({"fail": True}, max_retries=3)
        print(f"   Success after retries: {result8}")
    except Exception as e:
        print(f"   Failed after all retries: {e}")
    
    print("\n✅ All examples completed!")
    print("\nNote: In a real application, traces would be sent to the CognitionFlow API.")
    print("Make sure to configure your API key and project ID before running.")

if __name__ == "__main__":
    main()
