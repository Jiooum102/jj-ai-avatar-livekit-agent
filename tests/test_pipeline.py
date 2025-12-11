"""Unit tests for pipeline integration."""

import asyncio
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


class TestTalkingFacePipeline:
    """Test TalkingFacePipeline class."""

    def test_pipeline_initialization(self, mock_settings: None) -> None:
        """Test pipeline creation."""
        pipeline = TalkingFacePipeline()

        assert pipeline.stream_manager is not None
        assert pipeline.rabbitmq_consumer is not None
        assert pipeline.tts_provider is not None
        assert pipeline.talking_face_provider is not None
        assert not pipeline._running

    def test_pipeline_initialization_with_mocks(
        self, mock_settings: None
    ) -> None:
        """Test pipeline creation with mocked components."""
        mock_stream_manager = MagicMock()
        mock_consumer = MagicMock()
        mock_tts = MagicMock()
        mock_talking_face = MagicMock()

        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            rabbitmq_consumer=mock_consumer,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        assert pipeline.stream_manager is mock_stream_manager
        assert pipeline.rabbitmq_consumer is mock_consumer
        assert pipeline.tts_provider is mock_tts
        assert pipeline.talking_face_provider is mock_talking_face

    @pytest.mark.asyncio
    async def test_pipeline_start(self, mock_settings: None) -> None:
        """Test starting pipeline."""
        mock_stream_manager = AsyncMock()
        mock_consumer = AsyncMock()
        mock_tts = AsyncMock()
        mock_talking_face = AsyncMock()

        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            rabbitmq_consumer=mock_consumer,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        await pipeline.start()

        # Verify all components were initialized
        mock_tts.initialize.assert_called_once()
        mock_talking_face.initialize.assert_called_once()
        mock_stream_manager.start.assert_called_once()
        mock_consumer.start.assert_called_once()

        assert pipeline._running is True
        assert pipeline._processing_task is not None

        # Cleanup
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_stop(self, mock_settings: None) -> None:
        """Test stopping pipeline."""
        mock_stream_manager = AsyncMock()
        mock_consumer = AsyncMock()
        mock_tts = AsyncMock()
        mock_talking_face = AsyncMock()

        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            rabbitmq_consumer=mock_consumer,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        pipeline._running = True
        pipeline._processing_task = asyncio.create_task(asyncio.sleep(100))

        await pipeline.stop()

        # Verify cleanup
        mock_consumer.stop.assert_called_once()
        mock_stream_manager.stop.assert_called_once()
        mock_talking_face.cleanup.assert_called_once()
        mock_tts.cleanup.assert_called_once()

        assert pipeline._running is False

    @pytest.mark.asyncio
    async def test_pipeline_handle_message(self, mock_settings: None) -> None:
        """Test message handling."""
        mock_stream_manager = MagicMock()
        mock_stream_manager.state_manager = MagicMock()
        mock_stream_manager.state_manager.state = StreamState.IDLE

        pipeline = TalkingFacePipeline(stream_manager=mock_stream_manager)

        message = TextMessage(
            type="text",
            data="Hello, world!",
            session_id="test-session",
        )

        # Message should be queued
        await pipeline._handle_message(message)

        # Verify message was queued
        assert pipeline._message_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_pipeline_process_message_workflow(
        self, mock_settings: None
    ) -> None:
        """Test complete message processing workflow."""
        # Mock components
        mock_stream_manager = MagicMock()
        mock_stream_manager.state_manager = MagicMock()
        mock_stream_manager.state_manager.state = StreamState.IDLE
        mock_stream_manager.switch_to_talking_face = AsyncMock()
        mock_stream_manager.switch_to_static_video = AsyncMock()

        mock_tts = AsyncMock()
        mock_tts.synthesize = AsyncMock(return_value=b"fake audio bytes")

        mock_talking_face = AsyncMock()

        async def mock_frame_generator():
            """Mock frame generator."""
            for i in range(3):
                yield np.zeros((720, 1280, 3), dtype=np.uint8)

        mock_talking_face.generate_from_audio = AsyncMock(
            return_value=mock_frame_generator()
        )

        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        # Mock settings
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

            message = TextMessage(
                type="text",
                data="Hello, world!",
                session_id="test-session",
            )

            # Process message
            await pipeline._process_message(message)

            # Verify workflow
            mock_tts.synthesize.assert_called_once()
            mock_talking_face.generate_from_audio.assert_called_once()
            mock_stream_manager.switch_to_talking_face.assert_called_once()
            mock_stream_manager.switch_to_static_video.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_process_message_wrong_state(
        self, mock_settings: None
    ) -> None:
        """Test message processing when in wrong state."""
        mock_stream_manager = MagicMock()
        mock_stream_manager.state_manager = MagicMock()
        mock_stream_manager.state_manager.state = StreamState.PROCESSING

        pipeline = TalkingFacePipeline(stream_manager=mock_stream_manager)

        message = TextMessage(
            type="text",
            data="Hello, world!",
            session_id="test-session",
        )

        # Process message (should be re-queued)
        await pipeline._process_message(message)

        # Verify message was re-queued
        assert pipeline._message_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_pipeline_message_processing_loop(
        self, mock_settings: None
    ) -> None:
        """Test message processing loop."""
        mock_stream_manager = MagicMock()
        mock_stream_manager.state_manager = MagicMock()
        mock_stream_manager.state_manager.state = StreamState.IDLE
        mock_stream_manager.switch_to_talking_face = AsyncMock()
        mock_stream_manager.switch_to_static_video = AsyncMock()

        mock_tts = AsyncMock()
        mock_tts.synthesize = AsyncMock(return_value=b"fake audio bytes")

        mock_talking_face = AsyncMock()

        async def mock_frame_generator():
            """Mock frame generator."""
            yield np.zeros((720, 1280, 3), dtype=np.uint8)

        mock_talking_face.generate_from_audio = AsyncMock(
            return_value=mock_frame_generator()
        )

        pipeline = TalkingFacePipeline(
            stream_manager=mock_stream_manager,
            tts_provider=mock_tts,
            talking_face_provider=mock_talking_face,
        )

        # Mock settings
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

            pipeline._running = True

            # Add message to queue
            message = TextMessage(
                type="text",
                data="Hello, world!",
                session_id="test-session",
            )
            await pipeline._message_queue.put(message)

            # Start processing loop (run for a short time)
            task = asyncio.create_task(pipeline._message_processing_loop())
            await asyncio.sleep(0.5)
            pipeline._running = False
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify message was processed
            assert pipeline._message_queue.qsize() == 0

    def test_pipeline_get_statistics(self, mock_settings: None) -> None:
        """Test getting pipeline statistics."""
        mock_stream_manager = MagicMock()
        mock_stream_manager.get_statistics = MagicMock(
            return_value={"frames": 100}
        )

        pipeline = TalkingFacePipeline(stream_manager=mock_stream_manager)
        pipeline._running = True

        stats = pipeline.get_statistics()

        assert stats["running"] is True
        assert "message_queue_size" in stats
        assert "stream_manager" in stats

