# PoC Plan: Talking Face Livestream (RabbitMQ → RTMP)

## Overview

This is a Proof of Concept (PoC) demo that:
1. **Starts streaming immediately** with a static/default "not talking" video to RTMP
2. Consumes messages from RabbitMQ (text or audio)
3. When a message is received, generates talking face video from the input
4. Seamlessly transitions from static video to talking face video
5. After talking face completes, returns to static video
6. Maintains continuous RTMP stream throughout

## Architecture

```
Application Start
    ↓
Static Video Generator (Default/Idle State)
    ↓
FFmpeg Pipeline → RTMP Server (Continuous Stream)
    ↑
    |
RabbitMQ Queue → Consumer → Talking Face Generator
    (When message arrives, switch to talking face)
    (After completion, return to static video)
```

**Stream States:**
- **IDLE**: Streaming static/default video (no talking)
- **PROCESSING**: Generating talking face from RabbitMQ message
- **TALKING**: Streaming generated talking face video
- **TRANSITIONING**: Switching between static and talking video

## Implementation Plan

### Phase 1: Project Setup & Dependencies

#### 1.1 Update Requirements
- **File**: `requirements.txt`
- **Tasks**:
  - Add `pika` or `aio-pika` for RabbitMQ async consumer
  - Add `opencv-python` (cv2) for video frame processing
  - Add `numpy` for array operations
  - Add `requests` or `httpx` for API calls (if using external talking face API)
  - Add `pydantic` for message validation
  - Add `python-dotenv` for configuration
  - Add `asyncio` compatible libraries
  - **For Text-to-Speech (TTS)**:
    - `edge-tts` (recommended: Windows Edge TTS, free, high quality)
    - `pyttsx3` (alternative: cross-platform, lower quality)
    - `pydub` (for audio format conversion)
    - For API-based TTS: `openai` (for OpenAI TTS), `elevenlabs` (for ElevenLabs API)
  - **For Local AI Inference** (choose based on selected model):
    - `torch` and `torchvision` (PyTorch for model inference)
    - `torchaudio` (for audio processing)
    - `face-alignment` or `dlib` (for face detection if needed)
    - `librosa` and `soundfile` (for audio processing)
    - `Pillow` (for image processing)
    - `scipy` (for scientific computing)
    - `scikit-image` (for image processing utilities)
    - Model-specific packages (install via git or custom setup)

#### 1.2 Configuration
- **File**: `src/config/settings.py`
- **Tasks**:
  - RabbitMQ connection settings (host, port, queue name, credentials)
  - RTMP output URL configuration
  - **TTS settings**:
    - TTS provider type (local or API)
    - Local TTS settings (edge-tts voice selection, language)
    - API TTS settings (API key, endpoint, voice ID)
    - Audio output format (sample rate, channels, format)
  - Talking face API settings (if using external service)
  - **Local AI model settings** (Primary: MuseTalk):
    - Model type: MuseTalk (primary for PoC)
    - MuseTalk checkpoint paths
    - GPU/CPU device selection
    - Inference batch size
    - MuseTalk-specific parameters
    - Backup model settings (MimicTalk, SyncTalk) - for post-PoC evaluation
  - Video encoding parameters (resolution, fps, bitrate)
  - FFmpeg path configuration
  - Static video settings (default image/video path, loop behavior)
  - Default avatar image path (for local inference)

#### 1.3 Environment File
- **File**: `.env.dev.example`
- **Tasks**:
  - Add RabbitMQ connection details
  - Add RTMP URL
  - Add talking face API credentials (if needed)
  - Add video encoding settings

### Phase 2: RabbitMQ Consumer

#### 2.1 Message Consumer

- **File**: `src/poc/rabbitmq_consumer.py`
- **Design**: Async context manager with automatic reconnection
- **Tasks**:
  - Implement async RabbitMQ consumer using `aio-pika`
  - Connect to RabbitMQ server with retry logic
  - Subscribe to specified queue
  - Handle message acknowledgment (ack on success, nack on failure)
  - Parse incoming messages (expect JSON with text data)
  - Handle connection errors and reconnection logic
  - Support graceful shutdown

#### 2.2 Message Models

- **File**: `src/poc/models.py`
- **Design**: Pydantic models for type safety and validation
- **Tasks**:
  - Define Pydantic models for message structure:
    - `TextMessage`: Contains text to convert to talking face (via TTS)
    - `ControlMessage`: For control commands (start/stop stream)
  - Message validation and parsing
  - **Note**: Only TextMessage is supported. Workflow is: Text → TTS → Audio → Talking Face

### Phase 3: Static Video Generation (Idle State)

#### 3.1 Static Video Generator

- **File**: `src/poc/static_video.py`
- **Design**: Async iterator that yields frames at consistent rate
- **Tasks**:
  - Generate continuous static/default video frames
  - Support static image (repeated frames) or looping video
  - Maintain frame rate consistency (e.g., 30 fps)
  - Handle transitions smoothly
  - Support configurable default image/video path
  - Generate frames on-demand for continuous streaming

### Phase 3.5: Text-to-Speech (TTS) Module

#### 3.5.1 TTS Base Interface
- **File**: `src/poc/tts/base.py`
- **Design**: Abstract base class defining TTS interface
- **Tasks**:
  - Define abstract base class for TTS providers
  - Methods:
    - `synthesize(text: str, language: str, voice_id: Optional[str]) -> bytes`
    - `synthesize_streaming(text: str, language: str, voice_id: Optional[str]) -> AsyncIterator[bytes]`
  - Support multiple TTS backends (local and API-based)
  - Return audio in consistent format (WAV, 44.1kHz, mono)
  - Handle language and voice selection
  - Support streaming audio generation for low latency

#### 3.5.2 Local TTS Implementation
- **File**: `src/poc/tts/local_tts.py`
- **Design**: Local TTS using edge-tts (Windows Edge TTS) or pyttsx3
- **Tasks**:
  - Implement local TTS using edge-tts (recommended for PoC):
    - Free, no API key required
    - High quality voices
    - Multiple languages supported
    - Low latency
  - Alternative: pyttsx3 (cross-platform, but lower quality)
  - Audio format conversion:
    - Convert to WAV format
    - Normalize sample rate (44.1kHz)
    - Convert to mono channel
    - Handle audio encoding/decoding
  - Voice selection and management:
    - List available voices
    - Select voice by ID or language
    - Cache voice metadata
  - Error handling:
    - Handle TTS generation failures
    - Fallback to default voice
    - Retry logic for transient errors

#### 3.5.3 API-Based TTS Implementation
- **File**: `src/poc/tts/api_tts.py`
- **Design**: API-based TTS using external services
- **Tasks**:
  - Support multiple TTS API providers:
    - **ElevenLabs**: High quality, natural voices
    - **Cartesia**: Real-time streaming TTS
    - **Google Cloud TTS**: Enterprise-grade
    - **Azure TTS**: Microsoft voices
  - Handle API authentication
  - Make async API calls
  - Support streaming audio if API supports it
  - Handle API rate limits and errors
  - Audio format conversion to standard format
  - Retry logic with exponential backoff

#### 3.5.4 TTS Factory
- **File**: `src/poc/tts/factory.py`
- **Tasks**:
  - Factory pattern to select TTS provider (local vs API)
  - Initialize provider based on configuration
  - Handle provider switching at runtime
  - Provider health checking

### Phase 4: Talking Face Generation

**Note**: Model selection and comparison details are documented in the [Model Selection](#model-selection) section at the end of this document. This phase focuses on implementation details and code patterns.

**Selected Model for PoC**: MuseTalk (see Model Selection section for details and alternatives)

#### 4.1 Talking Face Interface

- **File**: `src/poc/talking_face/base.py`
- **Design**: Abstract base class defining interface that all models must implement
- **Tasks**:
  - Define abstract base class for talking face generators
  - Methods:
    - `generate_from_audio(audio: bytes, avatar: Union[str, Path]) -> AsyncIterator[np.ndarray]`
    - `generate_video_to_video(source_video: Union[str, Path], audio: bytes) -> AsyncIterator[np.ndarray]`
  - Support multiple input types:
    - Audio-to-talking-face (primary method)
    - **Video-to-video** (video input + audio) - MuseTalk supports this
  - **Note**: Text-to-talking-face is handled by pipeline: Text → TTS → Audio → Talking Face
  - Return video with consistent frame rate and resolution
  - Handle both image and video inputs

#### 4.2 API-Based Implementation (Option 1)
- **File**: `src/poc/talking_face/api_provider.py`
- **Tasks**:
  - Implement talking face generation using external API:
    - Options: Hedra, Tavus, D-ID, or similar
  - Handle API authentication
  - Make async API calls
  - Download/stream generated video
  - Handle API errors and retries
  - Support streaming response if API supports it

#### 4.3 Local Implementation (Option 2) - AI Pipeline Inference

- **File**: `src/poc/talking_face/local_provider.py`
- **Design**: Strategy pattern with factory for model selection
- **Tasks**:
  - Implement local talking face generation using **MuseTalk (Primary Model)**
  - MuseTalk-specific implementation:
    - Load MuseTalk model and checkpoints
    - Support image/video input + audio input
    - Real-time inference pipeline (30+ FPS target)
    - Video-to-video generation capability
  - Backup model support (for post-PoC evaluation):
    - Architecture to support MimicTalk and SyncTalk if needed
    - Model factory pattern for easy switching
  - Model loading and initialization:
    - Download pre-trained models on first run (or use local models)
    - Load models into GPU/CPU memory
    - Initialize face detection models
    - Cache models for faster subsequent inference
  - Audio preprocessing:
    - Convert input audio to required format (WAV, sample rate, etc.)
    - Extract audio features if needed by model
    - Handle different audio formats from RabbitMQ
  - Face preprocessing:
    - Load default avatar image
    - Face detection and alignment
    - Image preprocessing (resize, normalize, etc.)
  - Inference pipeline:
    - Run model inference (GPU-accelerated)
    - Generate video frames in batches or streaming
    - Handle frame rate consistency
    - Optimize for low latency
  - Post-processing:
    - Convert model output to standard video format
    - Ensure consistent resolution and frame rate
    - Handle color space conversion (RGB, BGR, etc.)
  - Performance optimization:
    - Model quantization (if supported)
    - Batch processing for multiple frames
    - GPU memory management
    - Async inference to avoid blocking
  - Error handling:
    - Handle model loading failures
    - Graceful degradation (fallback to static video)
    - Retry logic for inference errors

#### 4.3.1 Model-Specific Implementations

- **File**: `src/poc/talking_face/models/musetalk.py` (PRIMARY - Required for PoC)
- **Design**: Template method pattern - each model implements the same interface with model-specific logic
- **Tasks**:
  - **MuseTalk-specific implementation (Primary Model)**
  - Real-time inference optimization (30+ FPS target)
  - Streaming frame generation
  - Low-latency pipeline
  - Support for image/video input + audio input
  - Video-to-video generation capability

- **File**: `src/poc/talking_face/models/mimictalk.py` (BACKUP - Post-PoC)
- **Design**: Similar pattern, different preprocessing/inference logic
- **Tasks**:
  - MimicTalk-specific implementation (backup option)
  - 3D avatar generation
  - Per-identity training support

- **File**: `src/poc/talking_face/models/synctalk.py` (BACKUP - Post-PoC)
- **Design**: Similar pattern, Gaussian Splatting-specific rendering
- **Tasks**:
  - SyncTalk-specific implementation (backup option)
  - Gaussian Splatting rendering
  - Ultra-high speed optimization (101 FPS target)

#### 4.3.2 Model Manager
- **File**: `src/poc/talking_face/model_manager.py`
- **Tasks**:
  - Model download and caching
  - Model version management
  - Checkpoint loading
  - Model switching at runtime
  - Resource cleanup

#### 4.4 Talking Face Factory
- **File**: `src/poc/talking_face/factory.py`
- **Tasks**:
  - Factory pattern to select provider (API vs Local)
  - Initialize provider based on configuration
  - Handle provider switching

### Phase 5: RTMP Streaming

#### 5.1 FFmpeg Streamer

- **File**: `src/poc/rtmp_streamer.py`
- **Design**: Async subprocess management with stdin pipe for frame input
- **Tasks**:
  - Implement FFmpeg subprocess management
  - Create FFmpeg pipeline for RTMP streaming:
    - Input: Video frames from pipe (continuous)
    - Encoding: H.264 video, AAC audio
    - Output: RTMP stream (continuous, never stops)
  - Handle FFmpeg process lifecycle
  - Start streaming immediately on application start
  - Monitor stream health
  - Handle reconnection if stream drops (auto-reconnect)
  - Support different video formats and resolutions
  - Accept frames from different sources (static or talking face)

#### 5.2 Video Frame Buffer

- **File**: `src/poc/video_buffer.py`
- **Design**: Thread-safe async queue with frame rate control
- **Tasks**:
  - Implement frame buffer/queue for video frames
  - Handle frame rate synchronization (maintain consistent fps)
  - Manage buffer size to prevent memory issues
  - Support frame dropping if buffer overflows
  - Thread-safe operations
  - Support seamless source switching (static ↔ talking face)

### Phase 6: Pipeline Integration

#### 6.1 Stream State Manager
- **File**: `src/poc/stream_state.py`
- **Tasks**:
  - Define stream states: IDLE, PROCESSING, TALKING, TRANSITIONING
  - Manage state transitions
  - Thread-safe state management
  - State change callbacks/notifications

#### 6.2 Main Pipeline
- **File**: `src/poc/pipeline.py`
- **Tasks**:
  - Orchestrate the complete pipeline:
    1. **On Start**: Begin streaming static video immediately
    2. **On RabbitMQ Message** (TextMessage):
       - Switch state to PROCESSING
       - Convert text to audio using TTS module
       - Generate talking face video from audio
       - Switch state to TALKING
       - Stream talking face video
    3. **After Talking Face Completes**:
       - Switch state back to IDLE
       - Return to streaming static video
    4. **Workflow**: Text → TTS → Audio → Talking Face Video
  - Handle async operations
  - Manage seamless transitions between static and talking video
  - Ensure no gaps in RTMP stream
  - Coordinate TTS and talking face generation
  - Error handling and recovery
  - Logging and monitoring

#### 6.3 Stream Manager
- **File**: `src/poc/stream_manager.py`
- **Tasks**:
  - Manage continuous RTMP stream lifecycle
  - Start streaming immediately on application start (static video)
  - Handle seamless source switching:
    - Static video → Talking face video
    - Talking face video → Static video
  - Maintain stream continuity (no interruptions)
  - Handle multiple sequential messages
  - Buffer management between messages
  - Coordinate static video generator and talking face generator
  - Ensure frame rate consistency across transitions

### Phase 7: Main Application

#### 7.1 Entry Point
- **File**: `src/poc/main.py`
- **Tasks**:
  - Initialize configuration
  - Set up logging
  - Initialize static video generator
  - Initialize RTMP streamer (start streaming static video immediately)
  - Start RabbitMQ consumer (in background)
  - Initialize talking face provider
  - Start pipeline with initial static video streaming
  - Handle graceful shutdown (SIGINT, SIGTERM)
  - CLI argument parsing
  - Ensure stream starts before RabbitMQ consumer is ready

### Phase 8: Testing & Demo

#### 8.1 Unit Tests
- **Files**: `tests/poc/test_*.py`
- **Tasks**:
  - Test RabbitMQ consumer (mock RabbitMQ)
  - Test talking face generation (mock API/local)
  - Test RTMP streamer (mock FFmpeg)
  - Test message parsing
  - Test pipeline integration

#### 8.2 Integration Test
- **File**: `tests/poc/test_integration.py`
- **Tasks**:
  - Test end-to-end with mock RabbitMQ
  - Test with local FFmpeg (no actual RTMP server needed)
  - Verify message flow

#### 8.3 Demo Script
- **File**: `scripts/demo_poc.sh` or `scripts/demo_poc.py`
- **Tasks**:
  - Script to send test messages to RabbitMQ
  - Verify the PoC works end-to-end
  - Include instructions for setup

## Technical Details

### Message Format

```json
{
  "type": "text",
  "data": "Hello, this is a test message for talking face generation.",
  "session_id": "unique-session-id",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Or for audio:
```json
{
  "type": "audio",
  "data": "base64-encoded-audio",
  "format": "wav",
  "session_id": "unique-session-id",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### FFmpeg Command Template

```bash
ffmpeg -re \
  -f rawvideo \
  -vcodec rawvideo \
  -s 1280x720 \
  -pix_fmt rgb24 \
  -r 30 \
  -i pipe:0 \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -b:v 2000k \
  -maxrate 2000k \
  -bufsize 4000k \
  -pix_fmt yuv420p \
  -g 50 \
  -c:a aac \
  -b:a 128k \
  -ar 44100 \
  -f flv \
  rtmp://your-server.com/live/stream_key
```

### Directory Structure

```
src/
├── poc/
│   ├── __init__.py
│   ├── main.py                 # PoC entry point
│   ├── pipeline.py             # Main pipeline
│   ├── stream_manager.py       # Stream management
│   ├── stream_state.py         # Stream state management
│   ├── rabbitmq_consumer.py    # RabbitMQ consumer
│   ├── rtmp_streamer.py        # RTMP streaming
│   ├── static_video.py         # Static/default video generator
│   ├── video_buffer.py         # Video frame buffer
│   ├── models.py               # Message models
│   ├── tts/                     # Text-to-Speech module
│   │   ├── __init__.py
│   │   ├── base.py             # TTS base interface
│   │   ├── local_tts.py        # Local TTS implementation (edge-tts)
│   │   ├── api_tts.py          # API-based TTS implementation
│   │   └── factory.py          # TTS factory pattern
│   └── talking_face/
│       ├── __init__.py
│       ├── base.py             # Base interface
│       ├── api_provider.py     # API-based implementation
│       ├── local_provider.py   # Local AI inference implementation
│       ├── model_manager.py     # Model loading and management
│       ├── factory.py          # Factory pattern
│       └── models/             # Model-specific implementations
│           ├── __init__.py
│           ├── musetalk.py    # MuseTalk implementation (PRIMARY - Required for PoC)
│           ├── mimictalk.py   # MimicTalk implementation (BACKUP - Post-PoC)
│           └── synctalk.py    # SyncTalk implementation (BACKUP - Post-PoC)
└── config/
    └── settings.py             # Configuration (shared)
```

## Implementation Options

### Option A: API-Based Talking Face (Faster to implement)
- **Pros**: Quick setup, high quality, no local GPU needed
- **Cons**: API costs, network dependency, potential latency
- **Providers**: Hedra, Tavus, D-ID, Synthesia
- **Best for**: Quick PoC validation

### Option B: Local Talking Face (More control) - SELECTED FOR PoC
- **Pros**: No API costs, full control, lower latency potential
- **Cons**: Requires GPU, more complex setup, model management
- **Primary Library**: **MuseTalk** (selected for PoC)
- **Backup Libraries**: MimicTalk, SyncTalk (for post-PoC evaluation)
- **Best for**: Production-ready solution

### Recommendation
**Use Option B with MuseTalk** for PoC:
- MuseTalk provides good balance of setup ease and real-time performance
- No training required (zero-shot inference)
- Supports video input + audio input
- Real-time capable (30+ FPS)
- Backup models (MimicTalk, SyncTalk) can be evaluated post-PoC if specific requirements emerge

## Configuration Example

```python
# .env.dev
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_QUEUE=talking_face_input
RABBITMQ_VHOST=/

RTMP_URL=rtmp://your-server.com/live/stream_key
RTMP_RESOLUTION=1280x720
RTMP_FPS=30
RTMP_BITRATE=2000k

STATIC_VIDEO_PATH=./assets/default_avatar.png  # or video file
STATIC_VIDEO_LOOP=true  # Loop static video if it's a video file

TTS_PROVIDER=local  # 'api' or 'local'
TTS_PROVIDER_LOCAL=edge-tts  # 'edge-tts' or 'pyttsx3'
TTS_VOICE=en-US-AriaNeural  # Voice ID for edge-tts (optional, auto-selects by language)
TTS_LANGUAGE=en  # Default language code
TTS_SAMPLE_RATE=44100  # Audio sample rate (Hz)
TTS_CHANNELS=1  # Audio channels (1=mono, 2=stereo)
# TTS_API_PROVIDER=elevenlabs  # 'elevenlabs', 'openai', 'cartesia'
# TTS_API_KEY=your-api-key
# TTS_API_VOICE_ID=voice-id

TALKING_FACE_PROVIDER=local  # 'api' or 'local'
TALKING_FACE_MODEL=musetalk  # Primary: musetalk (backup: mimictalk, synctalk)
TALKING_FACE_MODEL_PATH=./models/musetalk  # Path to MuseTalk model checkpoints
TALKING_FACE_DEVICE=cuda  # 'cuda' or 'cpu'
TALKING_FACE_AVATAR_IMAGE=./assets/avatar.png  # Default avatar image (for MuseTalk)
TALKING_FACE_AVATAR_VIDEO=./assets/avatar_video.mp4  # Optional: video input for MuseTalk
TALKING_FACE_BATCH_SIZE=1  # Inference batch size
TALKING_FACE_API_URL=https://api.example.com
TALKING_FACE_API_KEY=your-api-key
TALKING_FACE_AVATAR_ID=avatar-123

FFMPEG_PATH=ffmpeg  # or full path if not in PATH
```

## Success Criteria

1. **Functionality**:
   - **Start streaming immediately** with static/default video when application starts
   - Successfully consume messages from RabbitMQ
   - Generate talking face video from text/audio when message arrives
   - **Seamlessly transition** from static video to talking face video
   - **Return to static video** after talking face completes
   - Maintain **continuous RTMP stream** without interruptions
   - Handle multiple sequential messages with smooth transitions

2. **Performance**:
   - End-to-end latency <5s for PoC
   - Stable RTMP stream without drops
   - Handle at least 1 message per second

3. **Reliability**:
   - Graceful error handling
   - Reconnection logic for RabbitMQ
   - Stream recovery if RTMP connection drops

## Testing Strategy

1. **Local Testing**:
   - Use local RabbitMQ instance
   - Use local RTMP server (nginx-rtmp or similar)
   - Test with sample messages

2. **Integration Testing**:
   - Mock talking face API for faster testing
   - Use FFmpeg with test RTMP server
   - Verify message flow end-to-end

3. **Demo Preparation**:
   - Create sample message producer script
   - Document setup instructions
   - Record demo video

## Timeline

- **Phase 1**: 1-2 days (Setup & Dependencies)
  - Additional time if setting up CUDA/PyTorch for local AI
- **Phase 2**: 1 day (RabbitMQ Consumer)
- **Phase 3**: 1 day (Static Video Generation)
- **Phase 4**: 3-5 days (Talking Face Generation)
  - **API-based**: 1-2 days (faster)
  - **Local AI**: 3-5 days (model setup, integration, optimization)
    - Model setup and testing: 1-2 days
    - Integration: 1-2 days
    - Optimization: 1 day
- **Phase 5**: 1-2 days (RTMP Streaming)
- **Phase 6**: 2-3 days (Pipeline Integration - includes seamless transitions)
- **Phase 7**: 1 day (Main Application)
- **Phase 8**: 1-2 days (Testing & Demo)

**Total**: 
- **With API-based**: 11-16 days (includes TTS module)
- **With Local AI**: 13-20 days (includes TTS module and model setup/optimization)

## Next Steps

1. Review [Model Selection](#model-selection) section for model details and setup instructions
2. Set up development environment (CUDA, PyTorch, MuseTalk)
3. Set up RabbitMQ instance (local or cloud)
4. Set up RTMP server for testing
5. Begin Phase 1: Project Setup

## Implementation Flow

### Application Startup Sequence:
1. Load configuration
2. Initialize static video generator
3. **Start RTMP stream immediately** with static video
4. Initialize RabbitMQ consumer (background)
5. Initialize TTS module
6. Initialize talking face provider
7. Ready to process messages

### Message Processing Flow:
1. **IDLE State**: Continuously streaming static video
2. **Message Received**: RabbitMQ consumer receives TextMessage
3. **PROCESSING State**: 
   - Continue streaming static video
   - Convert text to audio using TTS module
   - Generate talking face from audio in background
4. **TALKING State**: 
   - Talking face generation complete
   - Seamlessly switch video source from static to talking face
   - Stream talking face video
5. **After Talking Face Completes**:
   - Seamlessly switch back to static video
   - Return to IDLE state
   - Ready for next message

**Workflow**: Text → TTS → Audio → Talking Face Video

### Key Implementation Details:
- **No Stream Interruption**: RTMP stream must never stop, even during transitions
- **Frame Rate Consistency**: Maintain same fps (e.g., 30fps) for static and talking video
- **Smooth Transitions**: Use frame buffering to ensure smooth switching
- **Background Processing**: Generate talking face while still streaming static video
- **Queue Management**: Handle multiple messages by queuing them if one is processing

## Notes

- This PoC focuses on **continuous streaming** with seamless transitions
- Stream starts immediately on application start (not waiting for messages)
- Static video provides "idle" state when no messages are being processed
- Error handling should be robust but can be simplified for PoC
- Performance optimization can come after PoC validation
- The PoC can later be integrated into the full LiveKit agent architecture

---

## Model Selection

This section contains detailed model comparison, selection criteria, and setup instructions. The implementation plan uses **MuseTalk** as the primary model for PoC.

### Primary Model for PoC

**✅ MuseTalk** - **SELECTED AS PRIMARY MODEL**
- Repository: `OpenTalker/MuseTalk`
- Setup: Easy (⭐⭐)
- Performance: 30+ FPS real-time
- Features: Supports video input + audio input, zero-shot inference (no training required)
- Best for: Quick PoC implementation

### Backup Models (For Post-PoC Evaluation)

- **MimicTalk**: For 3D personalized avatars (requires 15 min training per identity, image input only)
- **SyncTalk**: For ultra-high speed (101 FPS) and highest quality (requires training per identity, complex setup)

### Comprehensive Model Comparison Table

| Model | GitHub | Type | Input | Output | Performance (GPU) | GPU VRAM | Setup Difficulty | Video Input | Best For |
|-------|--------|------|-------|--------|-------------------|----------|------------------|-------------|----------|
| **MuseTalk** | `OpenTalker/MuseTalk` | Real-time lip sync | Image/Video + Audio | Real-time lip-synced video | 30+ FPS (V100), 10-15 FPS (consumer) | 4GB+ | ⭐⭐ Easy | ✅ Yes | Real-time streaming, low latency |
| **MimicTalk** | `yerfor/MimicTalk` | 3D talking face | Image + Audio | 3D talking face video | Variable (depends on training) | 6GB+ | ⭐⭐⭐ Medium | ❌ No (Image only) | Personalized 3D avatars, fast training |
| **SyncTalk** | `ZiqiaoPeng/SyncTalk` | Gaussian Splatting | Training data + Audio | High-fidelity talking head | Up to 101 FPS (RTX 4090) | 8GB+ | ⭐⭐⭐⭐ Hard | ⚠️ Requires training | High-fidelity, real-time Gaussian splatting |

### Detailed Model Characteristics

| Model | Key Pros | Key Cons | Community Support | Real-time Ready |
|-------|----------|---------|-------------------|-----------------|
| **MuseTalk** | Designed for real-time (30+ FPS), optimized for streaming, low latency | Newer model, less community support | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Excellent |
| **MimicTalk** | Fast training (15 min per identity), high-quality 3D output, personalized avatars | Requires training per identity, image input only (no video), newer model | ⭐⭐ Limited | ⭐⭐⭐ Good (after training) |
| **SyncTalk** | Very fast (up to 101 FPS), Gaussian splatting, high-fidelity, tri-plane hash representations | Requires training per identity, complex setup, high GPU requirements (RTX 4090 recommended) | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ Excellent (after training) |

### Video-to-Video Model Comparison (Audio-Driven Only)

| Model | Video Input | Audio Input | Speed (GPU) | Quality | Setup | Real-time | Best Use Case |
|-------|------------|-------------|-------------|---------|-------|-----------|---------------|
| **MuseTalk** | ✅ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Real-time video streaming |
| **MimicTalk** | ❌ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 3D personalized avatars (image input) |
| **SyncTalk** | ⚠️ Training data | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High-fidelity, ultra-fast (101 FPS) |

### Gaussian Splatting Models for Talking Face Generation

Gaussian Splatting is a state-of-the-art 3D rendering technique that represents scenes using 3D Gaussian primitives, enabling high-quality, real-time rendering. For talking face generation, Gaussian Splatting models offer:

- **Ultra-fast rendering**: Can achieve 100+ FPS on high-end GPUs
- **High-fidelity output**: Photorealistic quality with fine details
- **Real-time capability**: Optimized for interactive applications
- **Multi-view consistency**: Better 3D consistency than 2D-based methods

#### Available Gaussian Splatting Models with Open-Source Code

| Model | GitHub Repository | Technology | Input | Training Required | Performance | GPU Requirements | Customization |
|-------|------------------|------------|-------|-------------------|-------------|------------------|---------------|
| **SyncTalk** | `ZiqiaoPeng/SyncTalk` | 3D Gaussian Splatting + Tri-plane Hash | Training data + Audio | ✅ Yes (per identity) | Up to 101 FPS (RTX 4090) | 8GB+ VRAM (RTX 4090 recommended) | ✅ Full source code available |
| **Portrait3DGS** | Research/Experimental | 3D Gaussian Splatting | Image/Video + Audio | ⚠️ Varies | Variable | 6GB+ VRAM | ⚠️ Limited availability |
| **GSTalker** | Research/Experimental | 3D Gaussian Splatting | Image + Audio | ⚠️ Varies | Variable | 6GB+ VRAM | ⚠️ Limited availability |

**Note**: As of 2024, **SyncTalk** is the primary Gaussian Splatting model for talking face generation with confirmed open-source code and active development. Other Gaussian Splatting models are primarily research prototypes with limited public code availability.

#### SyncTalk - Detailed Analysis

**Repository**: https://github.com/ZiqiaoPeng/SyncTalk

**Key Features**:
- ✅ **Open-source**: Full codebase available on GitHub
- ✅ **Audio-driven**: Supports audio input for talking head generation
- ✅ **High-speed**: Up to 101 FPS on RTX 4090
- ✅ **High-fidelity**: Photorealistic output using Gaussian Splatting
- ✅ **Customizable**: Full source code allows customization
- ✅ **Multi-view consistent**: Better 3D consistency than 2D methods

**Technical Architecture**:
- **Rendering**: 3D Gaussian Splatting with tri-plane hash representations
- **Audio Processing**: ASR model integration (AVE model)
- **Facial Animation**: Audio-visual synchronization module
- **Head Stabilization**: Multi-view consistent head pose control
- **Optimization**: CUDA-accelerated rendering pipeline

**Customization Capabilities**:
- ✅ Model architecture modifications
- ✅ Rendering pipeline customization
- ✅ Audio processing integration
- ✅ Training data format adaptation
- ✅ Inference optimization
- ✅ Multi-view rendering control

**Setup Complexity**: ⭐⭐⭐⭐ (Hard)
- Requires PyTorch3D, custom CUDA extensions
- Multiple encoder dependencies (freqencoder, shencoder, gridencoder, raymarching)
- Training pipeline setup required
- GPU optimization needed for best performance

**Best For**:
- High-fidelity talking face generation
- Real-time applications requiring 60+ FPS
- Custom avatar creation with training data
- Research and development projects
- Production systems with GPU resources

**Limitations**:
- Requires training per identity (not zero-shot)
- Complex setup with multiple dependencies
- High GPU requirements (RTX 4090 recommended)
- Training time required per identity
- Not suitable for quick prototyping without pre-trained models

#### Comparison: Gaussian Splatting vs. Other Approaches

| Approach | Speed | Quality | Setup | Training | Real-time | Customization |
|----------|-------|---------|-------|----------|-----------|---------------|
| **Gaussian Splatting (SyncTalk)** | ⭐⭐⭐⭐⭐ (101 FPS) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ Hard | Required | ⭐⭐⭐⭐⭐ | ✅ Full code |
| **2D-based (MuseTalk)** | ⭐⭐⭐⭐ (30+ FPS) | ⭐⭐⭐⭐ | ⭐⭐ Easy | Not required | ⭐⭐⭐⭐⭐ | ✅ Full code |
| **3D Mesh (MimicTalk)** | ⭐⭐⭐ (Variable) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ Medium | Required | ⭐⭐⭐ | ✅ Full code |

### Technical Requirements

#### Hardware Requirements
- **Minimum**: NVIDIA GPU with 4GB VRAM (GTX 1060 or better)
- **Recommended**: NVIDIA GPU with 6-8GB VRAM (RTX 3060, RTX 3070, or better)
- **CPU**: Multi-core CPU (8+ cores recommended for preprocessing)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for models and dependencies

#### Software Dependencies
- Python 3.8+ (3.10+ recommended)
- PyTorch 1.12+ (with CUDA support for GPU)
- CUDA Toolkit (if using GPU)
- cuDNN (for GPU acceleration)
- FFmpeg (already in Dockerfile)
- Additional model-specific dependencies:
  - Face detection: `dlib`, `face-alignment`, or `mediapipe`
  - Image processing: `opencv-python`, `Pillow`
  - Audio processing: `librosa`, `soundfile`
  - Deep learning: `torch`, `torchvision`, `numpy`, `scipy`

### Model Setup Instructions

#### MuseTalk Setup (Supports Video Input)
```bash
# Clone repository
git clone https://github.com/OpenTalker/MuseTalk.git
cd MuseTalk

# Install dependencies
pip install -r requirements.txt

# Download models
# Follow repository instructions for model download

# Test inference (image + audio)
python inference.py --input_image <image> --input_audio <audio> --output <output>

# Test inference with video (video-to-video)
# MuseTalk supports video input for video-to-video generation
```

#### MimicTalk Setup (3D Personalized Avatars - Image Input Only)
```bash
# Clone repository
git clone https://github.com/yerfor/MimicTalk.git
cd MimicTalk

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
# Follow repository instructions

# Train personalized avatar (15 minutes per identity)
python train.py --identity_image <image> --audio_data <audio>

# Test inference (image + audio)
python inference.py --identity_image <image> --input_audio <audio> --output <output>

# ⚠️ NOTE: MimicTalk uses image input only (not video input)
# Requires training per identity (15 minutes)
# Generates high-quality 3D talking faces
```

#### SyncTalk Setup (High-Fidelity Gaussian Splatting - Requires Training)
```bash
# Clone repository
git clone https://github.com/ZiqiaoPeng/SyncTalk.git
cd SyncTalk

# Create conda environment
conda create -n synctalk python=3.8.8
conda activate synctalk

# Install dependencies
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113/download.html
pip install ./freqencoder
pip install ./shencoder
pip install ./gridencoder
pip install ./raymarching

# Prepare training data
# Place training data in data/ folder

# Train model (requires training per identity)
python main.py data/May --workspace model/trial_may -O --iters 60000 --asr_model ave
python main.py data/May --workspace model/trial_may -O --iters 100000 --finetune_lips --patch_size 64 --asr_model ave

# Test inference (audio-driven)
python main.py data/May --workspace model/trial_may -O --test --test_train --asr_model ave --portrait --aud ./demo/test.wav

# ⚠️ NOTE: SyncTalk requires training per identity (not direct video input)
# Uses Gaussian Splatting for rendering
# Can achieve up to 101 FPS on RTX 4090
# High GPU requirements (8GB+ VRAM, RTX 4090 recommended)
# Very high-fidelity output
```

