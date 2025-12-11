"""Message models for RabbitMQ consumer.

This module defines Pydantic models for type-safe message validation
and parsing from RabbitMQ queues.
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class BaseMessage(BaseModel):
    """Base message model with common fields."""

    session_id: str = Field(..., description="Unique session identifier")
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Message timestamp (UTC)"
    )


class TextMessage(BaseMessage):
    """Message containing text to convert to talking face.

    This message type is used when sending text that needs to be
    converted to speech (via TTS) and then to talking face video.
    The workflow is: Text → TTS → Audio → Talking Face Video
    """

    type: Literal["text"] = Field(default="text", description="Message type")
    data: str = Field(..., min_length=1, description="Text content to convert to talking face")
    language: Optional[str] = Field(default="en", description="Language code (e.g., 'en', 'es', 'fr')")
    voice_id: Optional[str] = Field(default=None, description="Optional voice ID for TTS")


class ControlMessage(BaseMessage):
    """Control message for stream management.

    This message type is used for control commands like
    starting or stopping the stream.
    """

    type: Literal["control"] = Field(default="control", description="Message type")
    command: Literal["start", "stop", "pause", "resume"] = Field(
        ..., description="Control command"
    )
    reason: Optional[str] = Field(default=None, description="Optional reason for the command")


class Message(BaseModel):
    """Union type for all message types.

    This model can parse any of the message types and return
    the appropriate typed model.
    """

    @classmethod
    def parse_json(cls, json_data: str | bytes | dict) -> TextMessage | ControlMessage:
        """Parse JSON data into appropriate message type.

        Args:
            json_data: JSON string, bytes, or dict to parse.

        Returns:
            TextMessage or ControlMessage based on the 'type' field.

        Raises:
            ValueError: If message type is unknown or data is invalid.
        """
        if isinstance(json_data, bytes):
            json_data = json_data.decode("utf-8")

        if isinstance(json_data, str):
            import json

            data = json.loads(json_data)
        else:
            data = json_data

        msg_type = data.get("type")
        if msg_type == "text":
            return TextMessage(**data)
        elif msg_type == "control":
            return ControlMessage(**data)
        else:
            raise ValueError(f"Unknown message type: {msg_type}. Supported types: 'text', 'control'")

