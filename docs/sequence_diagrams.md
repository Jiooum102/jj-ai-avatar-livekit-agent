# Sequence Diagrams: PoC Processing Pipeline

This document contains sequence diagrams illustrating the flow of the talking face livestream PoC application.

## 1. Application Startup Sequence

This diagram shows the initialization sequence when the application starts.

```mermaid
sequenceDiagram
    participant Main as Main Application
    participant Config as Configuration
    participant StaticVideo as Static Video Generator
    participant RTMP as RTMP Streamer
    participant RabbitMQ as RabbitMQ Consumer
    participant TTS as TTS Module
    participant TalkingFace as Talking Face Provider
    participant StateMgr as Stream State Manager

    Main->>Config: Load settings from .env
    Config-->>Main: Settings loaded
    
    Main->>StateMgr: Initialize state manager
    StateMgr-->>Main: State = IDLE
    
    Main->>StaticVideo: Initialize static video generator
    StaticVideo-->>Main: Generator ready
    
    Main->>RTMP: Initialize RTMP streamer
    RTMP-->>Main: Streamer ready
    
    Main->>RTMP: Start streaming (static video)
    RTMP->>StaticVideo: Request frames
    StaticVideo-->>RTMP: Video frames (30 fps)
    RTMP->>RTMP: Encode frames (H.264)
    RTMP->>RTMP: Stream to RTMP server (continuous)
    Note over RTMP: Stream starts immediately
    
    Main->>RabbitMQ: Initialize consumer
    RabbitMQ->>RabbitMQ: Connect to RabbitMQ server
    RabbitMQ->>RabbitMQ: Subscribe to queue
    RabbitMQ-->>Main: Consumer ready (background)
    
    Main->>TTS: Initialize TTS provider
    TTS-->>Main: TTS provider ready
    
    Main->>TalkingFace: Initialize talking face provider
    TalkingFace-->>Main: Provider ready
    
    Main->>StateMgr: Set state = IDLE
    Note over Main: Application ready to process messages
```

## 2. Message Processing Flow (IDLE → TALKING → IDLE)

This diagram shows the complete flow when a message is received from RabbitMQ.

```mermaid
sequenceDiagram
    participant RabbitMQ as RabbitMQ Consumer
    participant Pipeline as Main Pipeline
    participant StateMgr as Stream State Manager
    participant StaticVideo as Static Video Generator
    participant TTS as TTS Module
    participant TalkingFace as Talking Face Provider
    participant RTMP as RTMP Streamer
    participant Buffer as Video Frame Buffer

    Note over RabbitMQ,RTMP: State: IDLE (streaming static video)
    
    RabbitMQ->>RabbitMQ: Receive message from queue
    RabbitMQ->>RabbitMQ: Parse JSON message
    RabbitMQ->>RabbitMQ: Validate message (Pydantic)
    RabbitMQ->>Pipeline: Message received (TextMessage)
    
    Pipeline->>StateMgr: Get current state
    StateMgr-->>Pipeline: State = IDLE
    
    Pipeline->>StateMgr: Set state = PROCESSING
    StateMgr-->>Pipeline: State updated
    
    Note over StaticVideo,RTMP: Continue streaming static video
    
    Pipeline->>TTS: Synthesize text to audio (async)
    TTS->>TTS: Generate speech from text
    TTS-->>Pipeline: Audio bytes (WAV format)
    Pipeline->>TalkingFace: Generate from audio (async)
    TalkingFace->>TalkingFace: Generate talking face video
    
    Note over TalkingFace: Generating frames in background...
    Note over StaticVideo,RTMP: Still streaming static video
    
    TalkingFace-->>Pipeline: Talking face video ready (frames)
    
    Pipeline->>StateMgr: Set state = TRANSITIONING
    StateMgr-->>Pipeline: State updated
    
    Pipeline->>Buffer: Switch source to talking face
    Buffer->>Buffer: Queue talking face frames
    
    Pipeline->>StateMgr: Set state = TALKING
    StateMgr-->>Pipeline: State updated
    
    RTMP->>Buffer: Request next frame
    Buffer-->>RTMP: Talking face frame
    RTMP->>RTMP: Encode frame
    RTMP->>RTMP: Stream to RTMP server
    Note over RTMP: Seamless transition (no interruption)
    
    loop While talking face has frames
        RTMP->>Buffer: Request frame
        Buffer->>TalkingFace: Get next frame
        TalkingFace-->>Buffer: Video frame
        Buffer-->>RTMP: Frame
        RTMP->>RTMP: Encode and stream
    end
    
    Note over TalkingFace: All frames generated
    
    Pipeline->>StateMgr: Set state = TRANSITIONING
    StateMgr-->>Pipeline: State updated
    
    Pipeline->>Buffer: Switch source back to static video
    Buffer->>Buffer: Queue static video frames
    
    Pipeline->>StateMgr: Set state = IDLE
    StateMgr-->>Pipeline: State updated
    
    RTMP->>Buffer: Request next frame
    Buffer->>StaticVideo: Get static frame
    StaticVideo-->>Buffer: Static frame
    Buffer-->>RTMP: Frame
    RTMP->>RTMP: Encode and stream
    Note over RTMP: Back to static video (seamless)
    
    Note over RabbitMQ,RTMP: State: IDLE (ready for next message)
```

## 3. Complete Pipeline with Error Handling

This diagram shows the complete pipeline including error handling and reconnection logic.

```mermaid
sequenceDiagram
    participant RabbitMQ as RabbitMQ Consumer
    participant Pipeline as Main Pipeline
    participant StateMgr as Stream State Manager
    participant StaticVideo as Static Video Generator
    participant TalkingFace as Talking Face Provider
    participant RTMP as RTMP Streamer
    participant Buffer as Video Frame Buffer
    participant ErrorHandler as Error Handler

    Note over RabbitMQ,RTMP: Application Running (IDLE state)
    
    RabbitMQ->>RabbitMQ: Monitor connection
    alt Connection Lost
        RabbitMQ->>ErrorHandler: Connection error detected
        ErrorHandler->>RabbitMQ: Trigger reconnection
        RabbitMQ->>RabbitMQ: Retry connection (with backoff)
        alt Reconnection Success
            RabbitMQ->>RabbitMQ: Resume consuming
        else Reconnection Failed
            RabbitMQ->>ErrorHandler: Max retries exceeded
            ErrorHandler->>Pipeline: Notify error
            Pipeline->>StateMgr: Set error state
        end
    end
    
    RabbitMQ->>RabbitMQ: Receive message
    RabbitMQ->>RabbitMQ: Parse and validate
    
    alt Invalid Message
        RabbitMQ->>RabbitMQ: Nack message (no requeue)
        RabbitMQ->>ErrorHandler: Log error
    else Valid Message
        RabbitMQ->>Pipeline: Deliver message
        
        Pipeline->>StateMgr: Set state = PROCESSING
        
        Pipeline->>TTS: Synthesize text to audio
        alt TTS Success
            TTS-->>Pipeline: Audio bytes
            Pipeline->>TalkingFace: Generate talking face from audio
        else TTS Failed
            TTS->>ErrorHandler: TTS error
            ErrorHandler->>Pipeline: Notify error
            Pipeline->>StateMgr: Set state = IDLE
            Note over RTMP: Continue streaming static video
        end
        
        alt Generation Success
            TalkingFace-->>Pipeline: Video frames ready
            Pipeline->>StateMgr: Set state = TALKING
            Pipeline->>Buffer: Switch to talking face
            Buffer->>RTMP: Provide frames
            RTMP->>RTMP: Stream continuously
        else Generation Failed
            TalkingFace->>ErrorHandler: Generation error
            ErrorHandler->>Pipeline: Notify error
            Pipeline->>StateMgr: Set state = IDLE
            Pipeline->>Buffer: Keep static video
            Note over RTMP: Continue streaming static video
        end
        
        alt RTMP Stream Error
            RTMP->>ErrorHandler: Stream error
            ErrorHandler->>RTMP: Attempt reconnection
            alt Reconnection Success
                RTMP->>RTMP: Resume streaming
            else Reconnection Failed
                RTMP->>ErrorHandler: Stream lost
                ErrorHandler->>Pipeline: Critical error
                Pipeline->>StateMgr: Set error state
            end
        end
    end
```

## 4. Multiple Message Queue Handling

This diagram shows how the system handles multiple sequential messages.

```mermaid
sequenceDiagram
    participant RabbitMQ as RabbitMQ Consumer
    participant Pipeline as Main Pipeline
    participant Queue as Message Queue
    participant StateMgr as Stream State Manager
    participant TalkingFace as Talking Face Provider
    participant RTMP as RTMP Streamer

    Note over RabbitMQ,RTMP: State: IDLE
    
    RabbitMQ->>RabbitMQ: Receive Message 1
    RabbitMQ->>Pipeline: Deliver Message 1
    Pipeline->>StateMgr: Set state = PROCESSING
    Pipeline->>TalkingFace: Generate from Message 1
    
    Note over RabbitMQ: Message 2 arrives
    RabbitMQ->>RabbitMQ: Receive Message 2
    RabbitMQ->>Pipeline: Deliver Message 2
    Pipeline->>Queue: Queue Message 2 (waiting)
    
    Note over RabbitMQ: Message 3 arrives
    RabbitMQ->>RabbitMQ: Receive Message 3
    RabbitMQ->>Pipeline: Deliver Message 3
    Pipeline->>Queue: Queue Message 3 (waiting)
    
    TalkingFace-->>Pipeline: Message 1 complete
    Pipeline->>StateMgr: Set state = TALKING
    Pipeline->>RTMP: Stream Message 1 video
    
    Note over RTMP: Streaming Message 1...
    
    RTMP-->>Pipeline: Message 1 streaming complete
    Pipeline->>StateMgr: Set state = IDLE
    
    Pipeline->>Queue: Get next message
    Queue-->>Pipeline: Message 2
    Pipeline->>StateMgr: Set state = PROCESSING
    Pipeline->>TalkingFace: Generate from Message 2
    
    Note over Queue: Message 3 still queued
    
    TalkingFace-->>Pipeline: Message 2 complete
    Pipeline->>StateMgr: Set state = TALKING
    Pipeline->>RTMP: Stream Message 2 video
    
    Note over RTMP: Streaming Message 2...
    
    RTMP-->>Pipeline: Message 2 streaming complete
    Pipeline->>StateMgr: Set state = IDLE
    
    Pipeline->>Queue: Get next message
    Queue-->>Pipeline: Message 3
    Pipeline->>StateMgr: Set state = PROCESSING
    Pipeline->>TalkingFace: Generate from Message 3
    
    Note over RabbitMQ,RTMP: Process continues...
```

## 5. Stream State Transitions

This diagram shows the state machine and valid transitions.

```mermaid
stateDiagram-v2
    [*] --> IDLE: Application Start
    
    IDLE --> PROCESSING: Message Received
    PROCESSING --> TALKING: Generation Complete
    PROCESSING --> IDLE: Generation Failed
    
    TALKING --> IDLE: Video Complete
    TALKING --> PROCESSING: New Message (Queue)
    
    IDLE --> ERROR: Critical Error
    PROCESSING --> ERROR: Critical Error
    TALKING --> ERROR: Critical Error
    
    ERROR --> IDLE: Error Recovered
    
    note right of IDLE
        Streaming static video
        Ready for messages
    end note
    
    note right of PROCESSING
        Generating talking face
        Still streaming static video
    end note
    
    note right of TALKING
        Streaming talking face video
        No new processing
    end note
    
    note right of ERROR
        Error state
        Attempting recovery
    end note
```

## 6. RTMP Streaming Continuity

This diagram emphasizes the continuous nature of the RTMP stream.

```mermaid
sequenceDiagram
    participant StaticVideo as Static Video
    participant TalkingFace as Talking Face Video
    participant Buffer as Frame Buffer
    participant RTMP as RTMP Streamer
    participant Server as RTMP Server

    Note over StaticVideo,Server: Continuous Stream (No Interruptions)
    
    loop Continuous Streaming
        RTMP->>Buffer: Request frame (30 fps)
        
        alt State = IDLE
            Buffer->>StaticVideo: Get frame
            StaticVideo-->>Buffer: Static frame
        else State = TALKING
            Buffer->>TalkingFace: Get frame
            TalkingFace-->>Buffer: Talking frame
        else State = TRANSITIONING
            Buffer->>Buffer: Switch source
            Buffer-->>RTMP: Frame (from new source)
        end
        
        Buffer-->>RTMP: Frame
        RTMP->>RTMP: Encode (H.264)
        RTMP->>Server: Stream frame
        Note over Server: Continuous stream maintained
    end
    
    Note over StaticVideo,Server: Stream never stops, even during transitions
```

## Key Design Principles Illustrated

1. **Continuous Streaming**: RTMP stream never stops, even during state transitions
2. **Seamless Transitions**: Frame buffer ensures smooth switching between sources
3. **Background Processing**: Talking face generation happens while streaming static video
4. **Error Resilience**: Automatic reconnection and error recovery
5. **Message Queueing**: Multiple messages are queued and processed sequentially
6. **State Management**: Clear state machine with valid transitions

## Component Interactions Summary

- **Main Pipeline**: Orchestrates all components and manages flow
- **Stream State Manager**: Tracks current state (IDLE, PROCESSING, TALKING, ERROR)
- **RabbitMQ Consumer**: Receives and validates messages
- **Static Video Generator**: Provides idle state frames
- **TTS Module**: Converts text messages to audio (Text-to-Speech)
- **Talking Face Provider**: Generates talking face video from audio
- **Video Frame Buffer**: Manages frame queue and source switching
- **RTMP Streamer**: Encodes and streams frames continuously

## Text-to-Speech Integration

The TTS module is a critical component that bridges text messages and talking face generation:

1. **TextMessage Processing** (Only supported message type):
   - Pipeline receives TextMessage from RabbitMQ
   - Pipeline calls TTS module to synthesize text to audio
   - TTS returns audio bytes (WAV format, 44.1kHz, mono)
   - Pipeline passes audio to Talking Face Provider for lipsync inference
   - **Workflow**: Text → TTS → Audio → Talking Face Video

2. **TTS Providers**:
   - **Local TTS** (edge-tts): Free, high quality, no API key needed
   - **API TTS** (ElevenLabs, OpenAI, etc.): Premium quality, requires API key

