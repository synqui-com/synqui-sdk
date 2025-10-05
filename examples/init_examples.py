#!/usr/bin/env python3
"""Examples demonstrating the new vaquero.init() method with different modes."""

import vaquero

def demonstrate_init_usage():
    """Demonstrate different ways to use vaquero.init()."""

    print("🚀 Vaquero SDK init() Examples")
    print("=" * 40)

    # Example 1: Simple development setup (default)
    print("\n1. Simple development setup (default mode):")
    print("   vaquero.init(api_key='your-api-key')")
    print("   - Uses development preset")
    print("   - Captures inputs, outputs, errors, code")
    print("   - Debug logging enabled")
    print("   - Small batch size (10) for quick feedback")

    # Example 2: Explicit development mode
    print("\n2. Explicit development mode:")
    print("   vaquero.init(api_key='your-api-key', mode='development')")
    print("   - Same as default, but explicit")

    # Example 3: Production mode
    print("\n3. Production mode:")
    print("   vaquero.init(api_key='your-api-key', mode='production')")
    print("   - Optimized for production use")
    print("   - Does NOT capture inputs/outputs/code")
    print("   - Only captures errors and tokens")
    print("   - Larger batch size (100) for efficiency")
    print("   - Debug logging disabled")

    # Example 4: Custom configuration with overrides
    print("\n4. Custom configuration with overrides:")
    print("   vaquero.init(")
    print("       api_key='your-api-key',")
    print("       mode='development',")
    print("       project_id='custom-project',")
    print("       capture_inputs=False,")
    print("       batch_size=50")
    print("   )")
    print("   - Starts with development preset")
    print("   - Applies custom overrides")

    # Example 5: Custom endpoint
    print("\n5. Custom endpoint:")
    print("   vaquero.init(")
    print("       api_key='your-api-key',")
    print("       endpoint='https://custom.vaquero.app'")
    print("   )")
    print("   - Use custom Vaquero endpoint")

def show_mode_differences():
    """Show the differences between development and production modes."""
    print("\n📊 Mode Comparison:")
    print("=" * 40)

    from vaquero.config import MODE_PRESETS

    dev = MODE_PRESETS['development']
    prod = MODE_PRESETS['production']

    print(f"{'Setting':<25} {'Development':<12} {'Production':<12}")
    print("-" * 50)

    settings = [
        ('capture_inputs', 'Inputs', dev['capture_inputs'], prod['capture_inputs']),
        ('capture_outputs', 'Outputs', dev['capture_outputs'], prod['capture_outputs']),
        ('capture_errors', 'Errors', dev['capture_errors'], prod['capture_errors']),
        ('capture_code', 'Code', dev['capture_code'], prod['capture_code']),
        ('capture_tokens', 'Tokens', dev['capture_tokens'], prod['capture_tokens']),
        ('auto_instrument_llm', 'LLM Auto', dev['auto_instrument_llm'], prod['auto_instrument_llm']),
        ('capture_system_prompts', 'Prompts', dev['capture_system_prompts'], prod['capture_system_prompts']),
        ('detect_agent_frameworks', 'Frameworks', dev['detect_agent_frameworks'], prod['detect_agent_frameworks']),
        ('debug', 'Debug', dev['debug'], prod['debug']),
        ('batch_size', 'Batch Size', dev['batch_size'], prod['batch_size']),
        ('flush_interval', 'Flush (s)', dev['flush_interval'], prod['flush_interval']),
    ]

    for key, name, dev_val, prod_val in settings:
        print(f"{name:<25} {str(dev_val):<12} {str(prod_val):<12}")

if __name__ == "__main__":
    demonstrate_init_usage()
    show_mode_differences()

    print("\n✅ Examples completed!")
    print("\nTo use the new init() method in your code:")
    print("1. Replace vaquero.configure(...) with vaquero.init(api_key='...')")
    print("2. Use mode='development' for development (default)")
    print("3. Use mode='production' for production")
    print("4. Add overrides as needed for your use case")
