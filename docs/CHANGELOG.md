# Changelog

All notable changes to the Vaquero Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial SDK implementation with simplified `init()` API
- Core tracing functionality with decorators and context managers
- Batch processing system with configurable batch sizes
- Error handling and circuit breaker patterns
- Memory management and monitoring
- Automatic LLM instrumentation (OpenAI, Anthropic, etc.)
- System prompt and code capture capabilities
- Comprehensive test suite
- Documentation and examples
- Mode presets (development/production) for easy configuration

## [0.1.0] - 2024-01-01

### Added
- Initial release of Vaquero Python SDK with simplified `init()` API
- Basic tracing decorators for sync and async functions
- Manual span creation with context managers
- Batch processing with configurable batch sizes
- Circuit breaker pattern for API calls
- Memory management and garbage collection
- Comprehensive configuration system with mode presets
- Environment variable support
- Error handling and retry logic
- Performance monitoring capabilities
- Automatic LLM instrumentation (OpenAI, Anthropic, etc.)
- System prompt and code capture capabilities
- Framework integration examples (FastAPI, Django, Flask)
- Comprehensive documentation
- Usage examples and best practices

### Features
- **Tracing**: Automatic function tracing with decorators
- **Spans**: Manual span creation for complex workflows
- **Batch Processing**: Efficient batch transmission of trace data
- **Error Handling**: Robust error handling with circuit breakers
- **Memory Management**: Automatic memory monitoring and cleanup
- **Configuration**: Flexible configuration system with mode presets
- **Framework Integration**: Easy integration with popular Python frameworks
- **LLM Instrumentation**: Automatic tracing of LLM API calls
- **Performance**: Minimal performance impact on applications
- **Reliability**: Graceful degradation and retry mechanisms

### API
- `vaquero.init()` - Simplified SDK initialization
- `@vaquero.trace()` - Function tracing decorator
- `vaquero.span()` - Manual span context manager
- `vaquero.flush()` - Manual trace flushing
- `vaquero.shutdown()` - SDK shutdown
- `vaquero.get_stats()` - SDK statistics and metrics

### Configuration Options
- API key and project ID (with auto-provisioning)
- Batch size and flush interval
- Retry logic and timeouts
- Input/output/token capture settings
- Environment and debug settings
- Circuit breaker configuration
- LLM auto-instrumentation settings
- Mode presets (development/production)

### Supported Python Versions
- Python 3.8+
- Python 3.9
- Python 3.10
- Python 3.11

### Dependencies
- `aiohttp>=3.8.0` - Async HTTP client
- `psutil>=5.9.0` - System and process utilities

### Development Dependencies
- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async testing support
- `black>=22.0.0` - Code formatting
- `isort>=5.10.0` - Import sorting
- `mypy>=0.991` - Type checking

## [0.0.1] - 2024-01-01

### Added
- Initial project setup
- Basic project structure
- Core SDK architecture
- Initial documentation

---

## Version History

### 0.1.0 (Current)
- First stable release
- Complete SDK functionality
- Comprehensive documentation
- Framework integration examples
- Production-ready features

### 0.0.1 (Development)
- Initial development version
- Basic project structure
- Core architecture design

## Migration Guide

### From 0.0.1 to 0.1.0

No breaking changes. This is the first stable release with complete functionality.

## Future Releases

### Planned Features
- [ ] Distributed tracing support
- [ ] Custom metrics collection
- [ ] Advanced filtering and sampling
- [ ] Real-time trace streaming
- [ ] Custom exporters
- [ ] Performance profiling
- [ ] Memory leak detection
- [ ] Custom dashboards
- [ ] Alerting and notifications
- [ ] Multi-language support

### Roadmap
- **Q1 2024**: Distributed tracing and custom metrics
- **Q2 2024**: Advanced filtering and real-time streaming
- **Q3 2024**: Performance profiling and memory leak detection
- **Q4 2024**: Custom dashboards and alerting

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Support

- **Documentation**: [https://docs.vaquero.com](https://docs.vaquero.com)
- **GitHub**: [https://github.com/vaquero/vaquero-python](https://github.com/vaquero/vaquero-python)
- **Issues**: [https://github.com/vaquero/vaquero-python/issues](https://github.com/vaquero/vaquero-python/issues)
- **Email**: support@vaquero.app

## License

This SDK is licensed under the MIT License. See the LICENSE file for details.
