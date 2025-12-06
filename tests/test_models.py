"""Unit tests for message models."""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from src.poc.models import ControlMessage, Message, TextMessage

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestTextMessage:
    """Test TextMessage model."""

    def test_text_message_creation(self) -> None:
        """Test creating TextMessage with required fields."""
        message = TextMessage(
            type="text",
            data="Hello, world!",
            session_id="test-session-123",
        )

        assert message.type == "text"
        assert message.data == "Hello, world!"
        assert message.session_id == "test-session-123"
        assert message.language == "en"  # default
        assert message.voice_id is None  # default
        assert isinstance(message.timestamp, datetime)

    def test_text_message_validation(self) -> None:
        """Test TextMessage validation (min_length)."""
        # Empty text should fail
        with pytest.raises(Exception):  # Pydantic validation error
            TextMessage(
                type="text",
                data="",  # Empty string should fail min_length=1
                session_id="test-session",
            )

    def test_text_message_optional_fields(self) -> None:
        """Test TextMessage with optional fields."""
        message = TextMessage(
            type="text",
            data="Test message",
            session_id="test-session",
            language="es",
            voice_id="es-ES-ElviraNeural",
        )

        assert message.language == "es"
        assert message.voice_id == "es-ES-ElviraNeural"

    def test_text_message_timestamp(self) -> None:
        """Test automatic timestamp generation."""
        before = datetime.now(UTC)
        message = TextMessage(
            type="text",
            data="Test",
            session_id="test-session",
        )
        after = datetime.now(UTC)

        assert before <= message.timestamp <= after


class TestControlMessage:
    """Test ControlMessage model."""

    def test_control_message_creation(self) -> None:
        """Test creating ControlMessage."""
        message = ControlMessage(
            type="control",
            command="start",
            session_id="test-session-123",
        )

        assert message.type == "control"
        assert message.command == "start"
        assert message.session_id == "test-session-123"
        assert message.reason is None  # default

    def test_control_message_commands(self) -> None:
        """Test all control command types."""
        commands = ["start", "stop", "pause", "resume"]

        for cmd in commands:
            message = ControlMessage(
                type="control",
                command=cmd,  # type: ignore
                session_id="test-session",
            )
            assert message.command == cmd

    def test_control_message_with_reason(self) -> None:
        """Test ControlMessage with reason field."""
        message = ControlMessage(
            type="control",
            command="stop",
            session_id="test-session",
            reason="User requested stop",
        )

        assert message.reason == "User requested stop"


class TestMessageParser:
    """Test Message.parse_json() factory method."""

    def test_message_parse_json_text(self) -> None:
        """Test parsing JSON to TextMessage."""
        json_data = {
            "type": "text",
            "data": "Hello, world!",
            "session_id": "test-session",
            "language": "en",
        }

        message = Message.parse_json(json_data)

        assert isinstance(message, TextMessage)
        assert message.data == "Hello, world!"
        assert message.session_id == "test-session"

    def test_message_parse_json_text_from_string(self) -> None:
        """Test parsing JSON string to TextMessage."""
        json_string = json.dumps({
            "type": "text",
            "data": "Test message",
            "session_id": "test-session",
        })

        message = Message.parse_json(json_string)

        assert isinstance(message, TextMessage)
        assert message.data == "Test message"

    def test_message_parse_json_text_from_bytes(self) -> None:
        """Test parsing JSON bytes to TextMessage."""
        json_bytes = json.dumps({
            "type": "text",
            "data": "Test message",
            "session_id": "test-session",
        }).encode("utf-8")

        message = Message.parse_json(json_bytes)

        assert isinstance(message, TextMessage)
        assert message.data == "Test message"

    def test_message_parse_json_control(self) -> None:
        """Test parsing JSON to ControlMessage."""
        json_data = {
            "type": "control",
            "command": "start",
            "session_id": "test-session",
        }

        message = Message.parse_json(json_data)

        assert isinstance(message, ControlMessage)
        assert message.command == "start"

    def test_message_parse_json_invalid(self) -> None:
        """Test parsing invalid message type."""
        json_data = {
            "type": "invalid_type",
            "data": "test",
            "session_id": "test-session",
        }

        with pytest.raises(ValueError, match="Unknown message type"):
            Message.parse_json(json_data)

    def test_message_parse_json_missing_type(self) -> None:
        """Test parsing JSON without type field."""
        json_data = {
            "data": "test",
            "session_id": "test-session",
        }

        with pytest.raises(ValueError, match="Unknown message type"):
            Message.parse_json(json_data)

    def test_message_parse_json_missing_required_fields(self) -> None:
        """Test parsing JSON with missing required fields."""
        json_data = {
            "type": "text",
            # Missing "data" and "session_id"
        }

        with pytest.raises(Exception):  # Pydantic validation error
            Message.parse_json(json_data)

