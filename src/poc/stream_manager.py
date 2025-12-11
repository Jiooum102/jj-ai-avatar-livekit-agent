"""Stream manager for continuous RTMP streaming with seamless source switching.

This module manages the continuous RTMP stream lifecycle, handling seamless
switching between static video and talking face video sources while maintaining
stream continuity.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from src.config import get_settings
from src.poc.rtmp_streamer import RTMPStreamer, create_rtmp_streamer
from src.poc.static_video import StaticVideoGenerator
from src.poc.stream_state import StreamState, StreamStateManager
from src.poc.video_buffer import AsyncFrameSource, VideoFrameBuffer

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages continuous RTMP stream with seamless source switching.

    This manager coordinates the RTMP streamer, video buffer, static video
    generator, and talking face generator to maintain a continuous stream
    with seamless transitions between sources.
    """

    def __init__(
        self,
        rtmp_streamer: Optional[RTMPStreamer] = None,
        video_buffer: Optional[VideoFrameBuffer] = None,
        state_manager: Optional[StreamStateManager] = None,
    ) -> None:
        """Initialize stream manager.

        Args:
            rtmp_streamer: RTMP streamer instance. If None, creates from settings.
            video_buffer: Video frame buffer instance. If None, creates from settings.
            state_manager: Stream state manager instance. If None, creates new one.
        """
        self.settings = get_settings()
        self.rtmp_settings = self.settings.rtmp

        # Initialize components
        self.rtmp_streamer = rtmp_streamer or create_rtmp_streamer(self.rtmp_settings)
        self.video_buffer = video_buffer or VideoFrameBuffer(
            max_size=self.rtmp_settings.fps * 4,  # 4 seconds of frames
            fps=self.rtmp_settings.fps,
        )
        self.state_manager = state_manager or StreamStateManager()

        # Static video generator
        self._static_video_gen: Optional[StaticVideoGenerator] = None
        self._static_video_source: Optional[AsyncFrameSource] = None

        # Talking face generator (set externally)
        self._talking_face_source: Optional[AsyncFrameSource] = None

        # Streaming control
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_streaming = False

        # Statistics
        self._frames_streamed = 0
        self._stream_start_time: Optional[float] = None

    async def start(self) -> None:
        """Start streaming with static video.

        This initializes the static video generator and starts the RTMP stream.
        """
        if self._is_streaming:
            logger.warning("Stream manager already started")
            return

        logger.info("Starting stream manager...")

        # Initialize static video generator
        self._static_video_gen = StaticVideoGenerator()
        await self._static_video_gen.__aenter__()

        # Create async frame source for static video
        self._static_video_source = AsyncFrameSource(self.video_buffer, "static")
        await self._static_video_source.start(self._static_video_gen)

        # Start RTMP streamer
        self.rtmp_streamer.start()

        # Start streaming thread
        self._stop_event.clear()
        self._is_streaming = True
        self._stream_start_time = time.time()
        self._streaming_thread = threading.Thread(target=self._streaming_loop, name="streaming", daemon=True)
        self._streaming_thread.start()

        # Set initial state
        self.state_manager.set_state(StreamState.IDLE)

        logger.info("Stream manager started")

    def stop(self) -> None:
        """Stop streaming and cleanup resources."""
        if not self._is_streaming:
            return

        logger.info("Stopping stream manager...")

        # Signal stop
        self._stop_event.set()
        self._is_streaming = False

        # Stop async sources
        if self._static_video_source:
            asyncio.run(self._static_video_source.stop())
            self._static_video_source = None

        if self._talking_face_source:
            asyncio.run(self._talking_face_source.stop())
            self._talking_face_source = None

        # Close static video generator
        if self._static_video_gen:
            asyncio.run(self._static_video_gen.__aexit__(None, None, None))
            self._static_video_gen = None

        # Stop RTMP streamer
        self.rtmp_streamer.stop()

        # Stop video buffer
        self.video_buffer.stop()

        # Wait for streaming thread
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=5)

        logger.info("Stream manager stopped")

    def _streaming_loop(self) -> None:
        """Main streaming loop that reads from buffer and pushes to RTMP."""
        frame_interval = 1.0 / self.rtmp_settings.fps
        last_frame_time = time.time()

        try:
            while not self._stop_event.is_set():
                # Get frame from buffer
                result = self.video_buffer.get_frame(timeout=0.1)
                if result is None:
                    # Buffer empty, generate silent audio and use last frame or black frame
                    # For now, just continue
                    time.sleep(0.01)
                    continue

                frame, audio = result

                # Maintain frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

                # Prepare audio (default to silence if None)
                if audio is None:
                    # Generate silence for one frame duration
                    samples_per_frame = int(self.rtmp_settings.audio_sample_rate / self.rtmp_settings.fps)
                    audio = np.zeros(samples_per_frame, dtype=np.int16)

                # Push to RTMP
                try:
                    self.rtmp_streamer.push(frame, audio)
                    self._frames_streamed += 1
                    last_frame_time = time.time()
                except Exception as e:
                    logger.error(f"Error pushing frame to RTMP: {e}")
                    # Set error state
                    self.state_manager.set_state(StreamState.ERROR, force=True)

        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
            self.state_manager.set_state(StreamState.ERROR, force=True)

    async def switch_to_talking_face(
        self,
        frame_generator: AsyncIterator[np.ndarray],
        audio_generator: Optional[AsyncIterator[np.ndarray]] = None,
    ) -> None:
        """Switch stream source to talking face video.

        Args:
            frame_generator: Async iterator yielding talking face video frames.
            audio_generator: Optional async iterator yielding audio chunks synchronized with frames.
        """
        logger.info("Switching to talking face source")

        # Set state to transitioning
        self.state_manager.set_state(StreamState.TRANSITIONING)

        # Stop static video source
        if self._static_video_source:
            await self._static_video_source.stop()
            self._static_video_source = None

        # Create and start talking face source
        self._talking_face_source = AsyncFrameSource(self.video_buffer, "talking_face")
        await self._talking_face_source.start(frame_generator, audio_generator)

        # Set state to talking
        self.state_manager.set_state(StreamState.TALKING)

        logger.info("Switched to talking face source")

    async def switch_to_static_video(self) -> None:
        """Switch stream source back to static video."""
        logger.info("Switching back to static video source")

        # Set state to transitioning
        self.state_manager.set_state(StreamState.TRANSITIONING)

        # Stop talking face source
        if self._talking_face_source:
            await self._talking_face_source.stop()
            self._talking_face_source = None

        # Restart static video source
        if self._static_video_gen is None:
            self._static_video_gen = StaticVideoGenerator()
            await self._static_video_gen.__aenter__()

        self._static_video_source = AsyncFrameSource(self.video_buffer, "static")
        await self._static_video_source.start(self._static_video_gen)

        # Set state to idle
        self.state_manager.set_state(StreamState.IDLE)

        logger.info("Switched back to static video source")

    def get_statistics(self) -> dict:
        """Get streaming statistics.

        Returns:
            Dictionary with streaming statistics.
        """
        uptime = time.time() - self._stream_start_time if self._stream_start_time else 0
        buffer_stats = self.video_buffer.get_statistics()

        return {
            "is_streaming": self._is_streaming,
            "frames_streamed": self._frames_streamed,
            "uptime_seconds": uptime,
            "current_state": self.state_manager.state.value,
            "buffer": buffer_stats,
        }

