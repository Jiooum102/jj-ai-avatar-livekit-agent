# LiveKit Live Avatar Agent

An SDK to integrate speech-to-text, LLM, and real-time talking face streaming for LiveKit.

## Project Description

The LiveKit Live Avatar Agent is a Python SDK designed to create intelligent, interactive avatars that can:
- Convert speech to text in real-time
- Process conversations using Large Language Models (LLMs)
- Generate and stream real-time talking face animations
- Integrate seamlessly with LiveKit for real-time communication

This SDK enables developers to build AI-powered avatar applications with natural conversation capabilities and lifelike facial animations.

## Features

- **Speech-to-Text**: Real-time audio transcription capabilities
- **LLM Integration**: Connect with Large Language Models for intelligent conversation processing
- **Real-time Talking Face Streaming**: Generate and stream animated facial expressions synchronized with speech
- **LiveKit Integration**: Built for LiveKit's real-time communication platform
- **Docker Support**: Containerized deployment with Docker
- **Development Tools**: Pre-configured development environment with linting and testing

## Architecture/Components

The SDK is structured to handle the following components:

1. **Audio Processing**: Captures and processes audio streams from LiveKit
2. **Speech Recognition**: Converts audio to text using speech-to-text services
3. **LLM Processing**: Processes text through language models to generate responses
4. **Avatar Rendering**: Generates talking face animations based on speech output
5. **Streaming**: Sends video/audio streams back to LiveKit participants

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized deployment)
- FFmpeg (required for audio/video processing)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd jj-ai-avatar-livekit-agent
```

2. Install dependencies:
```bash
make dev_install
```

Or manually:
```bash
pip install -r requirements.txt
pre-commit install
```

3. Set up environment variables:
```bash
cp .env.dev.example .env.dev  # If example exists
# Edit .env.dev with your configuration
```

## Usage

### Running the Agent

```bash
make dev_run_server
```

Or using the development script directly:
```bash
./scripts/develop.sh run
```

### Docker Deployment

Build and run with Docker:
```bash
docker build -t livekit-avatar-agent .
docker run -e <your-env-vars> livekit-avatar-agent
```

## Development

### Development Setup

1. Set up the development environment:
```bash
make dev_install
make dev_setup_env
```

2. Run tests:
```bash
make dev_test
```

3. Run linting:
```bash
make dev_test_lint
```

4. Fix linting issues:
```bash
make dev_fix_lint
```

### Available Make Targets

- `dev_install`: Install project dependencies and pre-commit hooks
- `dev_setup_env`: Set up the development environment
- `dev_run_server`: Run the development server
- `dev_test`: Run unit tests
- `dev_test_lint`: Run linting checks
- `dev_fix_lint`: Auto-fix linting issues

### Code Quality

The project uses:
- **Black**: Code formatting (line length: 120)
- **isort**: Import sorting
- **flake8**: Linting
- **pytest**: Testing framework
- **pre-commit**: Git hooks for automatic linting

## Requirements

### Python Dependencies

See [requirements.txt](requirements.txt) for the complete list of Python packages.

Key dependencies include:
- Python 3.8+
- Black, isort, flake8 (code quality)
- pytest (testing)
- pre-commit (git hooks)

### System Requirements

- FFmpeg (for audio/video processing)
- Docker (optional, for containerized deployment)

## Project Structure

```
jj-ai-avatar-livekit-agent/
├── Dockerfile              # Docker container configuration
├── Makefile                # Development commands
├── pyproject.toml          # Python project configuration
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── scripts/
│   ├── develop.sh          # Development script
│   └── run_tests.sh        # Test runner script
├── src/
│   ├── __init__.py
│   └── main.py             # Main application entry point
└── tests/
    └── test_example.py     # Example test file
```

## Contributing

1. Set up your development environment (see Development section)
2. Create a feature branch
3. Make your changes
4. Ensure tests pass: `make dev_test`
5. Ensure linting passes: `make dev_test_lint`
6. Submit a pull request
