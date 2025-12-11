"""Unit tests for RTMP streamer (with mocked FFmpeg)."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.poc.rtmp_streamer import RTMPStreamer, create_rtmp_streamer

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestRTMPStreamer:
    """Test RTMPStreamer class."""

    def test_streamer_initialization(self, mock_settings: None) -> None:
        """Test streamer creation."""
        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
        )

        assert streamer.url == "rtmp://test.example.com/live/test"
        assert streamer.width == 1280
        assert streamer.height == 720
        assert streamer.fps == 30
        assert streamer.process is None

    def test_streamer_initialization_with_custom_params(self, mock_settings: None) -> None:
        """Test streamer creation with custom parameters."""
        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1920,
            height=1080,
            fps=60,
            video_bitrate="5000k",
            audio_bitrate="192k",
            audio_sample_rate=48000,
            preset="fast",
            tune="zerolatency",
        )

        assert streamer.width == 1920
        assert streamer.height == 1080
        assert streamer.fps == 60
        assert streamer.video_bitrate == "5000k"
        assert streamer.audio_bitrate == "192k"
        assert streamer.audio_sample_rate == 48000
        assert streamer.preset == "fast"
        assert streamer.tune == "zerolatency"

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    @patch("os.open")
    @patch("os.fdopen")
    def test_streamer_start(
        self,
        mock_fdopen: MagicMock,
        mock_open: MagicMock,
        mock_mkfifo: MagicMock,
        mock_popen: MagicMock,
        mock_settings: None,
    ) -> None:
        """Test starting RTMP streamer."""
        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process

        # Mock FIFO file descriptors
        mock_fd = MagicMock()
        mock_open.return_value = mock_fd
        mock_file = MagicMock()
        mock_file.fileno.return_value = mock_fd
        mock_fdopen.return_value = mock_file

        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
            data_dir=tempfile.mkdtemp(),
        )

        streamer.start()

        # Verify FIFOs were created
        assert mock_mkfifo.call_count == 2

        # Verify FFmpeg process was started
        mock_popen.assert_called_once()
        assert streamer.process is not None

        # Verify writer threads were started
        assert streamer._video_thread is not None
        assert streamer._audio_thread is not None
        assert streamer._video_thread.is_alive()
        assert streamer._audio_thread.is_alive()

        # Cleanup
        streamer.stop()

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    @patch("os.open")
    @patch("os.fdopen")
    def test_streamer_push_frame(
        self,
        mock_fdopen: MagicMock,
        mock_open: MagicMock,
        mock_mkfifo: MagicMock,
        mock_popen: MagicMock,
        mock_settings: None,
    ) -> None:
        """Test pushing frame and audio to streamer."""
        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process

        # Mock FIFO file descriptors
        mock_fd = MagicMock()
        mock_open.return_value = mock_fd
        mock_file = MagicMock()
        mock_file.fileno.return_value = mock_fd
        mock_fdopen.return_value = mock_file

        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
            data_dir=tempfile.mkdtemp(),
        )

        streamer.start()

        # Create test frame and audio
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        audio = np.zeros(1470, dtype=np.int16)  # ~1 frame of audio at 44.1kHz, 30fps

        # Push frame and audio
        streamer.push(frame, audio)

        # Give threads time to process
        import time

        time.sleep(0.1)

        # Verify queues have items (or were processed)
        # The queues might be empty if processed quickly, so we just verify no errors
        assert streamer.process is not None

        # Cleanup
        streamer.stop()

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    @patch("os.open")
    @patch("os.fdopen")
    def test_streamer_push_frame_wrong_size(
        self,
        mock_fdopen: MagicMock,
        mock_open: MagicMock,
        mock_mkfifo: MagicMock,
        mock_popen: MagicMock,
        mock_settings: None,
    ) -> None:
        """Test pushing frame with wrong size (should be resized)."""
        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process

        # Mock FIFO file descriptors
        mock_fd = MagicMock()
        mock_open.return_value = mock_fd
        mock_file = MagicMock()
        mock_file.fileno.return_value = mock_fd
        mock_fdopen.return_value = mock_file

        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
            data_dir=tempfile.mkdtemp(),
        )

        streamer.start()

        # Create frame with wrong size (should be resized)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        audio = np.zeros(1470, dtype=np.int16)

        # Should not raise error (resizing happens in writer thread)
        streamer.push(frame, audio)

        import time

        time.sleep(0.1)

        # Cleanup
        streamer.stop()

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    def test_streamer_start_ffmpeg_failure(
        self, mock_mkfifo: MagicMock, mock_popen: MagicMock, mock_settings: None
    ) -> None:
        """Test handling of FFmpeg startup failure."""
        # Mock FFmpeg process failure
        mock_popen.side_effect = Exception("FFmpeg not found")

        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
            data_dir=tempfile.mkdtemp(),
        )

        with pytest.raises(Exception):
            streamer.start()

        # Cleanup should handle partial initialization
        streamer.stop()

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    @patch("os.open")
    @patch("os.fdopen")
    def test_streamer_push_when_not_started(
        self,
        mock_fdopen: MagicMock,
        mock_open: MagicMock,
        mock_mkfifo: MagicMock,
        mock_popen: MagicMock,
        mock_settings: None,
    ) -> None:
        """Test pushing frame when streamer is not started."""
        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
        )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        audio = np.zeros(1470, dtype=np.int16)

        with pytest.raises(RuntimeError, match="not initialized"):
            streamer.push(frame, audio)

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    @patch("os.open")
    @patch("os.fdopen")
    def test_streamer_stop(
        self,
        mock_fdopen: MagicMock,
        mock_open: MagicMock,
        mock_mkfifo: MagicMock,
        mock_popen: MagicMock,
        mock_settings: None,
    ) -> None:
        """Test stopping RTMP streamer."""
        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        # Mock FIFO file descriptors
        mock_fd = MagicMock()
        mock_open.return_value = mock_fd
        mock_file = MagicMock()
        mock_file.fileno.return_value = mock_fd
        mock_file.close = MagicMock()
        mock_fdopen.return_value = mock_file

        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
            data_dir=tempfile.mkdtemp(),
        )

        streamer.start()
        assert streamer.process is not None

        streamer.stop()

        # Verify process was terminated
        mock_process.terminate.assert_called_once()
        # Verify file was closed
        assert mock_file.close.call_count == 2  # Video and audio FIFOs

    def test_create_rtmp_streamer_from_settings(self, mock_settings: None) -> None:
        """Test creating streamer from settings."""
        from src.config import get_settings

        settings = get_settings()
        streamer = create_rtmp_streamer(settings.rtmp)

        assert streamer.url == settings.rtmp.url
        assert streamer.width == settings.rtmp.width
        assert streamer.height == settings.rtmp.height
        assert streamer.fps == settings.rtmp.fps

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    @patch("os.open")
    @patch("os.fdopen")
    def test_streamer_audio_format_conversion(
        self,
        mock_fdopen: MagicMock,
        mock_open: MagicMock,
        mock_mkfifo: MagicMock,
        mock_popen: MagicMock,
        mock_settings: None,
    ) -> None:
        """Test audio format conversion (float to int16)."""
        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process

        # Mock FIFO file descriptors
        mock_fd = MagicMock()
        mock_open.return_value = mock_fd
        mock_file = MagicMock()
        mock_file.fileno.return_value = mock_fd
        mock_fdopen.return_value = mock_file

        streamer = RTMPStreamer(
            url="rtmp://test.example.com/live/test",
            width=1280,
            height=720,
            fps=30,
            data_dir=tempfile.mkdtemp(),
        )

        streamer.start()

        # Create frame and float audio (should be converted to int16)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        audio_float = np.array([0.5, -0.5, 0.0], dtype=np.float32)

        # Should not raise error (conversion happens in writer thread)
        streamer.push(frame, audio_float)

        import time

        time.sleep(0.1)

        # Cleanup
        streamer.stop()

