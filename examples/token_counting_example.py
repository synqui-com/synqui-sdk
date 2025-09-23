"""Example demonstrating automatic token counting in CognitionFlow SDK.

This example shows how the SDK automatically counts tokens for LLM calls
and regular function calls, providing accurate cost estimation and
performance monitoring.
"""

import asyncio
import cognitionflow
from typing import Dict, Any, List


# Configure the SDK
cognitionflow.configure(
    api_key="your-api-key",
    project_id="your-project-id",
    capture_tokens=True  # Enable automatic token counting
)


@cognitionflow.trace("llm_agent")
async def call_llm(prompt: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Simulate an LLM call with automatic token counting."""
    # Simulate LLM response with usage information
    response = {
        "content": f"Generated response for: {prompt[:50]}...",
        "model": model,
        "usage": {
            "prompt_tokens": len(prompt.split()) * 1.3,  # Approximate token count
            "completion_tokens": 150,
            "total_tokens": len(prompt.split()) * 1.3 + 150
        }
    }
    
    # The SDK will automatically extract token counts from the response
    return response


@cognitionflow.trace("data_processor")
def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process data with automatic token counting for inputs/outputs."""
    # Simulate data processing
    processed_items = []
    for item in data:
        processed_items.append({
            "id": item.get("id"),
            "processed": True,
            "value": item.get("value", 0) * 2
        })
    
    return {
        "processed_count": len(processed_items),
        "items": processed_items,
        "summary": f"Processed {len(processed_items)} items successfully"
    }


@cognitionflow.trace("research_agent")
async def research_topic(topic: str, depth: str = "shallow") -> Dict[str, Any]:
    """Research a topic with multiple LLM calls."""
    # First LLM call
    research_prompt = f"Research the topic: {topic}"
    research_result = await call_llm(research_prompt)
    
    # Second LLM call for deeper analysis
    if depth == "deep":
        analysis_prompt = f"Analyze this research: {research_result['content']}"
        analysis_result = await call_llm(analysis_prompt)
        
        return {
            "topic": topic,
            "research": research_result,
            "analysis": analysis_result,
            "depth": depth
        }
    
    return {
        "topic": topic,
        "research": research_result,
        "depth": depth
    }


async def main():
    """Main example function."""
    print("🚀 CognitionFlow Token Counting Example")
    print("=" * 50)
    
    # Example 1: Simple LLM call
    print("\n1. Simple LLM Call:")
    result1 = await call_llm("What is artificial intelligence?")
    print(f"   Response: {result1['content'][:100]}...")
    print(f"   Model: {result1['model']}")
    print(f"   Tokens: {result1['usage']['total_tokens']}")
    
    # Example 2: Data processing
    print("\n2. Data Processing:")
    sample_data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30}
    ]
    result2 = process_data(sample_data)
    print(f"   Processed {result2['processed_count']} items")
    print(f"   Summary: {result2['summary']}")
    
    # Example 3: Complex workflow with multiple LLM calls
    print("\n3. Research Workflow:")
    result3 = await research_topic("machine learning", "deep")
    print(f"   Topic: {result3['topic']}")
    print(f"   Research tokens: {result3['research']['usage']['total_tokens']}")
    print(f"   Analysis tokens: {result3['analysis']['usage']['total_tokens']}")
    
    # Flush any pending traces
    cognitionflow.flush()
    print("\n✅ All traces sent to CognitionFlow!")


if __name__ == "__main__":
    asyncio.run(main())
