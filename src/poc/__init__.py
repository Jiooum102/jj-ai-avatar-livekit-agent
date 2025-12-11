"""PoC (Proof of Concept) module for talking face livestream."""

# Lazy imports to avoid circular dependencies
# Import only lightweight models directly
from src.poc.models import ControlMessage, Message, TextMessage

__all__ = [
    "Message",
    "TextMessage",
    "ControlMessage",
    # Other exports available via direct imports
    # "StaticVideoGenerator",
    # "VideoFrameBuffer",
    # "AsyncFrameSource",
    # "StreamState",
    # "StreamStateManager",
    # "StreamManager",
    # "TalkingFacePipeline",
    # "RabbitMQConsumer",
    # "RTMPStreamer",
    # "create_rtmp_streamer",
]

