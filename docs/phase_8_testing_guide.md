# Phase 8: Testing & Demo Guide

This document provides instructions for running tests and using the demo script for the PoC talking face livestream application.

## Overview

Phase 8 includes:
- **Unit Tests**: Test individual components (RTMP streamer, pipeline, etc.)
- **Integration Tests**: Test end-to-end message flow
- **Demo Script**: Send test messages to RabbitMQ

## Prerequisites

1. **RabbitMQ** must be running locally on port 5672 (default)
2. **Python dependencies** installed (`pip install -r requirements.txt`)
3. **Test data** available in `data/tests/` directory

## Running Tests

### Run All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Suites

```bash
# Run unit tests only
pytest tests/test_rtmp_streamer.py
pytest tests/test_pipeline.py
pytest tests/test_rabbitmq_consumer.py
pytest tests/test_talking_face.py
pytest tests/test_static_video.py

# Run integration tests
pytest tests/test_integration.py -v

# Run with integration marker
pytest tests/ -m integration
```

### Test Files

- **`test_rtmp_streamer.py`**: Tests for RTMP streaming with mocked FFmpeg
- **`test_pipeline.py`**: Tests for pipeline integration
- **`test_integration.py`**: End-to-end integration tests
- **`test_rabbitmq_consumer.py`**: RabbitMQ consumer tests (already exists)
- **`test_talking_face.py`**: Talking face provider tests (already exists)
- **`test_static_video.py`**: Static video generator tests (already exists)

## Demo Script

The demo script (`scripts/demo_poc.py`) sends test messages to RabbitMQ to verify the end-to-end pipeline.

### Basic Usage

```bash
# Send default test messages
python scripts/demo_poc.py

# Send a custom message
python scripts/demo_poc.py --message "Hello, world!"

# Send multiple messages
python scripts/demo_poc.py --messages "First" "Second" "Third"

# Interactive mode
python scripts/demo_poc.py --interactive

# Send messages from JSON file
python scripts/demo_poc.py --file data/tests/example_messages.json
```

### Command-Line Options

- `--message, -m`: Send a single message
- `--messages`: Send multiple messages (space-separated)
- `--file, -f`: Send messages from a JSON file
- `--interactive, -i`: Interactive mode for sending messages
- `--delay`: Delay between messages in seconds (default: 2.0)
- `--queue`: Override RabbitMQ queue name
- `--host`: Override RabbitMQ host
- `--port`: Override RabbitMQ port

### Example JSON File Format

```json
[
  {
    "type": "text",
    "data": "Hello! This is a test message.",
    "session_id": "test-session-1",
    "language": "en"
  },
  {
    "type": "text",
    "data": "This is another message.",
    "session_id": "test-session-2",
    "language": "en"
  }
]
```

Or simple string array:

```json
[
  "First message",
  "Second message",
  "Third message"
]
```

## Testing Workflow

### 1. Start RabbitMQ

Make sure RabbitMQ is running:

```bash
# Check if RabbitMQ is running
rabbitmqctl status

# Or start RabbitMQ (if installed locally)
sudo systemctl start rabbitmq-server
```

### 2. Run Unit Tests

```bash
# Verify all components work correctly
pytest tests/ -v
```

### 3. Run Integration Tests

```bash
# Test end-to-end flow (with mocked components)
pytest tests/test_integration.py -v
```

### 4. Test with Demo Script

```bash
# In one terminal, start the PoC application
python -m src.poc.main

# In another terminal, send test messages
python scripts/demo_poc.py
```

### 5. Interactive Testing

```bash
# Start the application
python -m src.poc.main

# In another terminal, use interactive mode
python scripts/demo_poc.py --interactive
```

## Expected Behavior

When running the demo script:

1. **Messages are sent** to RabbitMQ queue
2. **Application receives** messages from RabbitMQ
3. **Text is converted** to audio using TTS
4. **Talking face video** is generated from audio
5. **Stream switches** from static video to talking face
6. **After completion**, stream returns to static video

## Troubleshooting

### RabbitMQ Connection Issues

```bash
# Check RabbitMQ status
rabbitmqctl status

# Check if port 5672 is open
netstat -an | grep 5672

# Check RabbitMQ logs
tail -f /var/log/rabbitmq/rabbitmq.log
```

### Test Failures

- **Mock issues**: Some tests use mocks - verify mock setup
- **Missing test data**: Ensure `data/tests/` contains required files
- **Configuration**: Check `.env` file or environment variables

### Demo Script Issues

- **Connection refused**: Verify RabbitMQ is running on correct host/port
- **Queue not found**: Queue is auto-created, but check permissions
- **Message format**: Verify JSON format matches expected schema

## Success Criteria

Phase 8 is successful when:

1. ✅ All unit tests pass
2. ✅ Integration tests pass
3. ✅ Demo script successfully sends messages
4. ✅ End-to-end flow works (message → TTS → talking face → RTMP)
5. ✅ Multiple sequential messages are processed correctly
6. ✅ Error handling works gracefully

## Next Steps

After Phase 8 completion:

1. Review test coverage
2. Add additional test cases if needed
3. Document any issues found
4. Prepare for demo/presentation
5. Consider performance testing

## Notes

- Tests use mocks for external dependencies (FFmpeg, RabbitMQ, etc.)
- Integration tests verify the complete workflow
- Demo script requires actual RabbitMQ instance
- For production testing, use actual RTMP server

