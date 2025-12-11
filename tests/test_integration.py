"""Integration tests for end-to-end message flow.

These tests verify the complete pipeline from RabbitMQ message
to RTMP streaming, using mocked components where necessary.
"""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.poc.models import TextMessage
from src.poc.pipeline import TalkingFacePipeline
from src.poc.stream_state import StreamState

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for end-to-end pipeline."""

    @pytest.mark.asyncio
    async def test_message_flow_text_to_talking_face(
        self, mock_settings: None
    ) -> None:
        """Test complete flow: Text → TTS → Audio → Talking Face → RTMP."""
        # Mock all external dependencies
        mock_stream_manager = MagicMock()
        mock_stream_manager.state_manager = MagicMock()
        mock_stream_manager.state_manager.state = StreamState.IDLE
        mock_stream_manager.state_manager.set_state = MagicMock()
        mock_stream_manager.switch_to_talking_face = AsyncMock()
        mock_stream_manager.switch_to_static_video = AsyncMock()
        mock_stream_manager.start = AsyncMock()
        mock_stream_manager.stop = MagicMock()

        mock_consumer = AsyncMock()
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()

        # Mock TTS provider
        mock_tts = AsyncMock()
        # Create realistic audio bytes (WAV format)
        import soundfile as sf
        import io

        sample_rate = 44100
        duration = 1.0  # 1 second
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))).astype(
            np.float32
        )
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format="WAV")
        audio_bytes = audio_buffer.getvalue()

        mock_tts.synthesize = AsyncMock(return_value=audio_bytes)
        mock_tts.initialize = AsyncMock()
        mock_tts.cleanup = AsyncMock()

        # Mock talking face provider
        mock_talking_face = AsyncMock()

        async def mock_frame_generator():
            """Generate mock frames."""
            for i in range(30):  # 1 second at 30fps
                yield np.zeros((720, 1280, 3), dtype=np.uint8)

        mock_talking_face.generate_from_audio = AsyncMock(
            return_value=mock_frame_generator()
        )
        mock_talking_face.initialize = AsyncMock()
        mock_talking_face.cleanup = AsyncMock()

        # Create pipeline with mocks
        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            rabbitmq_consumer=mock_consumer,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        # Mock settings
        with patch("src.poc.pipeline.get_settings") as mock_get_settings:
            mock_settings_obj = MagicMock()
            mock_settings_obj.rtmp.fps = 30
            mock_settings_obj.rtmp.width = 1280
            mock_settings_obj.rtmp.height = 720
            mock_settings_obj.rtmp.audio_sample_rate = 44100
            mock_settings_obj.talking_face.musetalk.avatar_image = Path(
                "/tmp/test_avatar.png"
            )
            mock_get_settings.return_value = mock_settings_obj
            pipeline.settings = mock_settings_obj

            # Start pipeline
            await pipeline.start()

            # Simulate receiving a message
            message = TextMessage(
                type="text",
                data="Hello, this is a test message.",
                session_id="test-session-123",
                language="en",
            )

            # Process message
            await pipeline._process_message(message)

            # Verify complete workflow
            mock_tts.synthesize.assert_called_once_with(
                text="Hello, this is a test message.",
                language="en",
                voice_id=None,
            )
            mock_talking_face.generate_from_audio.assert_called_once()
            mock_stream_manager.switch_to_talking_face.assert_called_once()
            mock_stream_manager.switch_to_static_video.assert_called_once()

            # Cleanup
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_multiple_messages_sequential(
        self, mock_settings: None
    ) -> None:
        """Test processing multiple messages sequentially."""
        mock_stream_manager = MagicMock()
        mock_stream_manager.state_manager = MagicMock()
        mock_stream_manager.state_manager.state = StreamState.IDLE
        mock_stream_manager.switch_to_talking_face = AsyncMock()
        mock_stream_manager.switch_to_static_video = AsyncMock()
        mock_stream_manager.start = AsyncMock()
        mock_stream_manager.stop = MagicMock()

        mock_consumer = AsyncMock()
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()

        mock_tts = AsyncMock()

        import soundfile as sf
        import io

        sample_rate = 44100
        duration = 0.5
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))).astype(
            np.float32
        )
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format="WAV")
        audio_bytes = audio_buffer.getvalue()

        mock_tts.synthesize = AsyncMock(return_value=audio_bytes)
        mock_tts.initialize = AsyncMock()
        mock_tts.cleanup = AsyncMock()

        mock_talking_face = AsyncMock()

        async def mock_frame_generator():
            for i in range(15):  # 0.5 second at 30fps
                yield np.zeros((720, 1280, 3), dtype=np.uint8)

        mock_talking_face.generate_from_audio = AsyncMock(
            return_value=mock_frame_generator()
        )
        mock_talking_face.initialize = AsyncMock()
        mock_talking_face.cleanup = AsyncMock()

        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            rabbitmq_consumer=mock_consumer,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        with patch("src.poc.pipeline.get_settings") as mock_get_settings:
            mock_settings_obj = MagicMock()
            mock_settings_obj.rtmp.fps = 30
            mock_settings_obj.rtmp.width = 1280
            mock_settings_obj.rtmp.height = 720
            mock_settings_obj.rtmp.audio_sample_rate = 44100
            mock_settings_obj.talking_face.musetalk.avatar_image = Path(
                "/tmp/test_avatar.png"
            )
            mock_get_settings.return_value = mock_settings_obj
            pipeline.settings = mock_settings_obj

            await pipeline.start()

            # Process multiple messages
            messages = [
                TextMessage(type="text", data="First message", session_id="session-1"),
                TextMessage(type="text", data="Second message", session_id="session-2"),
                TextMessage(type="text", data="Third message", session_id="session-3"),
            ]

            for message in messages:
                await pipeline._process_message(message)

            # Verify all messages were processed
            assert mock_tts.synthesize.call_count == 3
            assert mock_stream_manager.switch_to_talking_face.call_count == 3
            assert mock_stream_manager.switch_to_static_video.call_count == 3

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_message_parsing_from_rabbitmq_format(
        self, mock_settings: None
    ) -> None:
        """Test parsing RabbitMQ message format."""
        from src.poc.models import Message

        # Simulate RabbitMQ message format
        rabbitmq_message = {
            "type": "text",
            "data": "Hello from RabbitMQ!",
            "session_id": "rabbitmq-session-123",
            "timestamp": "2024-01-01T00:00:00Z",
            "language": "en",
        }

        # Parse message
        parsed = Message.parse_json(rabbitmq_message)

        assert isinstance(parsed, TextMessage)
        assert parsed.type == "text"
        assert parsed.data == "Hello from RabbitMQ!"
        assert parsed.session_id == "rabbitmq-session-123"
        assert parsed.language == "en"

    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(
        self, mock_settings: None
    ) -> None:
        """Test error handling during message processing."""
        mock_stream_manager = MagicMock()
        mock_stream_manager.state_manager = MagicMock()
        mock_stream_manager.state_manager.state = StreamState.IDLE
        mock_stream_manager.state_manager.set_state = MagicMock()
        mock_stream_manager.switch_to_static_video = AsyncMock()
        mock_stream_manager.start = AsyncMock()
        mock_stream_manager.stop = MagicMock()

        mock_consumer = AsyncMock()
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()

        # Mock TTS to raise error
        mock_tts = AsyncMock()
        mock_tts.synthesize = AsyncMock(side_effect=Exception("TTS error"))
        mock_tts.initialize = AsyncMock()
        mock_tts.cleanup = AsyncMock()

        mock_talking_face = AsyncMock()
        mock_talking_face.initialize = AsyncMock()
        mock_talking_face.cleanup = AsyncMock()

        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            rabbitmq_consumer=mock_consumer,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        with patch("src.poc.pipeline.get_settings") as mock_get_settings:
            mock_settings_obj = MagicMock()
            mock_settings_obj.rtmp.fps = 30
            mock_settings_obj.rtmp.width = 1280
            mock_settings_obj.rtmp.height = 720
            mock_settings_obj.talking_face.musetalk.avatar_image = Path(
                "/tmp/test_avatar.png"
            )
            mock_get_settings.return_value = mock_settings_obj
            pipeline.settings = mock_settings_obj

            await pipeline.start()

            message = TextMessage(
                type="text",
                data="This will fail",
                session_id="error-session",
            )

            # Process message (should handle error gracefully)
            await pipeline._process_message(message)

            # Verify error state was set
            mock_stream_manager.state_manager.set_state.assert_called()
            # Verify recovery attempt
            mock_stream_manager.switch_to_static_video.assert_called_once()

            await pipeline.stop()

