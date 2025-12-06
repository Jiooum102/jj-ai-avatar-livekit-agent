"""PoC (Proof of Concept) module for talking face livestream."""

from src.poc.models import ControlMessage, Message, TextMessage
from src.poc.static_video import StaticVideoGenerator

# Lazy import for RabbitMQConsumer to avoid requiring aio_pika at import time
try:
    from src.poc.rabbitmq_consumer import RabbitMQConsumer

    __all__ = [
        "RabbitMQConsumer",
        "Message",
        "TextMessage",
        "ControlMessage",
        "StaticVideoGenerator",
    ]
except ImportError:
    # RabbitMQConsumer not available (aio_pika not installed)
    __all__ = [
        "Message",
        "TextMessage",
        "ControlMessage",
        "StaticVideoGenerator",
    ]

