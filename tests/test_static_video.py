"""Unit tests for static video generator."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.poc.static_video import StaticVideoGenerator

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestStaticVideoGenerator:
    """Test StaticVideoGenerator class."""

    def test_generator_initialization(self, mock_settings: None) -> None:
        """Test generator creation."""
        generator = StaticVideoGenerator(
            source_path=Path("test.png"),
            target_width=1280,
            target_height=720,
            fps=30,
        )

        assert generator.target_width == 1280
        assert generator.target_height == 720
        assert generator.fps == 30
        assert generator.frame_interval == 1.0 / 30

    @pytest.mark.asyncio
    async def test_generator_context_manager(
        self, test_image_path: Path, mock_settings: None
    ) -> None:
        """Test async context manager."""
        async with StaticVideoGenerator(source_path=test_image_path) as generator:
            assert generator._is_running is True
            assert generator._source_type == "image"
            assert generator._static_frame is not None

        assert generator._is_running is False

    @pytest.mark.asyncio
    async def test_generator_load_static_image(
        self, test_image_path: Path, mock_settings: None
    ) -> None:
        """Test loading static image file."""
        async with StaticVideoGenerator(source_path=test_image_path) as generator:
            assert generator._source_type == "image"
            assert generator._static_frame is not None
            assert isinstance(generator._static_frame, np.ndarray)
            # Verify RGB format (3 channels)
            assert len(generator._static_frame.shape) == 3
            assert generator._static_frame.shape[2] == 3

    @pytest.mark.asyncio
    async def test_generator_load_video_file(
        self, test_video_path: Path, mock_settings: None
    ) -> None:
        """Test loading video file."""
        async with StaticVideoGenerator(source_path=test_video_path) as generator:
            assert generator._source_type == "video"
            assert generator._video_cap is not None

    @pytest.mark.asyncio
    async def test_generator_image_repeat(
        self, test_image_path: Path, mock_settings: None
    ) -> None:
        """Test image frame repetition."""
        async with StaticVideoGenerator(
            source_path=test_image_path, fps=10  # Lower FPS for faster test
        ) as generator:
            frames = []
            async for frame in generator:
                frames.append(frame)
                if len(frames) >= 3:  # Get 3 frames
                    break

            # All frames should be the same (repeated image)
            assert len(frames) == 3
            assert np.array_equal(frames[0], frames[1])
            assert np.array_equal(frames[1], frames[2])

    @pytest.mark.asyncio
    async def test_generator_video_loop(
        self, test_video_path: Path, mock_settings: None
    ) -> None:
        """Test video looping behavior."""
        async with StaticVideoGenerator(
            source_path=test_video_path, loop=True, fps=30
        ) as generator:
            # Get a few frames to verify looping works
            frame_count = 0
            async for frame in generator:
                frame_count += 1
                assert isinstance(frame, np.ndarray)
                assert frame.shape[2] == 3  # RGB
                if frame_count >= 5:  # Test a few frames
                    break

            assert frame_count >= 5

    @pytest.mark.asyncio
    async def test_generator_frame_resize(
        self, test_image_path: Path, mock_settings: None
    ) -> None:
        """Test frame resizing to target resolution."""
        target_width, target_height = 640, 480

        async with StaticVideoGenerator(
            source_path=test_image_path,
            target_width=target_width,
            target_height=target_height,
            fps=10,
        ) as generator:
            frame = await generator.__anext__()

            assert frame.shape[0] == target_height
            assert frame.shape[1] == target_width
            assert frame.shape[2] == 3  # RGB channels

    @pytest.mark.asyncio
    async def test_generator_rgb_format(
        self, test_image_path: Path, mock_settings: None
    ) -> None:
        """Verify frames are in RGB format."""
        async with StaticVideoGenerator(
            source_path=test_image_path, fps=10
        ) as generator:
            frame = await generator.__anext__()

            # Verify numpy array
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8
            assert len(frame.shape) == 3
            assert frame.shape[2] == 3  # RGB channels
            # Verify values are in valid range
            assert frame.min() >= 0
            assert frame.max() <= 255

    @pytest.mark.asyncio
    async def test_generator_missing_file(self, mock_settings: None) -> None:
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            async with StaticVideoGenerator(
                source_path=Path("nonexistent_file.jpg")
            ):
                pass

    @pytest.mark.asyncio
    async def test_generator_unsupported_format(
        self, temp_dir: Path, mock_settings: None
    ) -> None:
        """Test unsupported file formats."""
        # Create a file with unsupported extension
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            async with StaticVideoGenerator(source_path=unsupported_file):
                pass

    @pytest.mark.asyncio
    async def test_generator_configuration_integration(
        self, test_image_path: Path, mock_settings: None
    ) -> None:
        """Test integration with configuration settings."""
        # Test with default settings (from config)
        # This will use settings from get_settings()
        async with StaticVideoGenerator() as generator:
            # Should use settings from config
            assert generator.target_width is not None
            assert generator.target_height is not None
            assert generator.fps is not None

    @pytest.mark.asyncio
    async def test_generator_frame_rate_timing(
        self, test_image_path: Path, mock_settings: None
    ) -> None:
        """Test frame rate consistency (timing)."""
        import time

        fps = 10  # Low FPS for testing
        expected_interval = 1.0 / fps

        async with StaticVideoGenerator(
            source_path=test_image_path, fps=fps
        ) as generator:
            start_time = time.time()
            frame1 = await generator.__anext__()
            time_after_first = time.time()

            frame2 = await generator.__anext__()
            time_after_second = time.time()

            # Verify frames were generated
            assert frame1 is not None
            assert frame2 is not None

            # Verify timing (should be approximately the frame interval)
            # Allow some tolerance for async overhead
            elapsed = time_after_second - time_after_first
            assert elapsed >= expected_interval * 0.8  # At least 80% of expected
            assert elapsed <= expected_interval * 2.0  # Not more than 2x expected

