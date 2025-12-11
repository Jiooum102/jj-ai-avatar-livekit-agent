"""Video frame buffer for seamless source switching.

This module provides a thread-safe frame buffer that manages video frames from
different sources (static video, talking face) and ensures seamless transitions
between them while maintaining consistent frame rate.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import AsyncIterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VideoFrameBuffer:
    """Thread-safe video frame buffer with source switching support.

    This buffer manages frames from different sources (static video, talking face)
    and provides seamless switching between them. It maintains frame rate
    synchronization and handles buffer overflow by dropping frames.

    Example:
        ```python
        buffer = VideoFrameBuffer(max_size=120, fps=30)

        # Start consuming from static video
        async for frame in static_video_generator:
            buffer.put_frame(frame)

        # Switch to talking face
        async for frame in talking_face_generator:
            buffer.put_frame(frame)
        ```
    """

    def __init__(
        self,
        max_size: int = 120,
        fps: int = 30,
        drop_on_overflow: bool = True,
    ) -> None:
        """Initialize video frame buffer.

        Args:
            max_size: Maximum number of frames to buffer. Defaults to 120 (4 seconds at 30fps).
            fps: Target frame rate. Defaults to 30.
            drop_on_overflow: If True, drop oldest frames when buffer is full. Defaults to True.
        """
        self.max_size = max_size
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.drop_on_overflow = drop_on_overflow

        # Thread-safe queue for frames
        self._frame_queue: Queue = Queue(maxsize=max_size)
        self._audio_queue: Queue = Queue(maxsize=max_size)

        # Frame metadata for synchronization
        self._frame_timestamps: deque = deque(maxlen=max_size)
        self._last_frame_time: Optional[float] = None

        # Source tracking
        self._current_source: Optional[str] = None
        self._source_lock = threading.Lock()

        # Statistics
        self._frames_dropped = 0
        self._frames_put = 0
        self._frames_get = 0

        # Control flags
        self._stopped = threading.Event()

    def put_frame(
        self,
        frame: np.ndarray,
        audio: Optional[np.ndarray] = None,
        source: Optional[str] = None,
    ) -> bool:
        """Put a frame into the buffer.

        Args:
            frame: Video frame as numpy array (RGB, uint8).
            audio: Optional audio chunk synchronized with frame. Defaults to None.
            source: Optional source identifier (e.g., 'static', 'talking_face'). Defaults to None.

        Returns:
            True if frame was added, False if dropped.
        """
        if self._stopped.is_set():
            return False

        current_time = time.time()

        # Update source tracking
        if source is not None:
            with self._source_lock:
                if self._current_source != source:
                    logger.debug(f"Source switched: {self._current_source} -> {source}")
                    self._current_source = source

        # Try to put frame
        try:
            self._frame_queue.put_nowait((frame, current_time))
            if audio is not None:
                self._audio_queue.put_nowait((audio, current_time))
            else:
                # Put None audio to maintain synchronization
                self._audio_queue.put_nowait((None, current_time))

            self._frame_timestamps.append(current_time)
            self._frames_put += 1
            return True

        except Full:
            if self.drop_on_overflow:
                # Drop oldest frame
                try:
                    self._frame_queue.get_nowait()
                    self._audio_queue.get_nowait()
                    if self._frame_timestamps:
                        self._frame_timestamps.popleft()
                    self._frames_dropped += 1

                    # Retry putting new frame
                    self._frame_queue.put_nowait((frame, current_time))
                    if audio is not None:
                        self._audio_queue.put_nowait((audio, current_time))
                    else:
                        self._audio_queue.put_nowait((None, current_time))

                    self._frame_timestamps.append(current_time)
                    self._frames_put += 1
                    logger.warning(f"Buffer full, dropped oldest frame. Total dropped: {self._frames_dropped}")
                    return True
                except (Full, Empty):
                    # Still full or empty after drop, give up
                    self._frames_dropped += 1
                    logger.warning(f"Failed to add frame after drop attempt. Total dropped: {self._frames_dropped}")
                    return False
            else:
                # Don't drop, just fail
                self._frames_dropped += 1
                logger.warning(f"Buffer full, frame dropped. Total dropped: {self._frames_dropped}")
                return False

    def get_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Get a frame from the buffer.

        Args:
            timeout: Maximum time to wait for a frame in seconds. If None, blocks indefinitely.

        Returns:
            Tuple of (frame, audio) or None if timeout/stopped.
        """
        if self._stopped.is_set():
            return None

        try:
            frame_data, frame_time = self._frame_queue.get(timeout=timeout)
            audio_data, _ = self._audio_queue.get(timeout=0.1)  # Should be synchronized

            self._frames_get += 1
            self._last_frame_time = frame_time

            return (frame_data, audio_data)

        except Empty:
            return None

    def get_frame_nowait(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Get a frame from the buffer without blocking.

        Returns:
            Tuple of (frame, audio) or None if buffer is empty.
        """
        return self.get_frame(timeout=0)

    def size(self) -> int:
        """Get current buffer size.

        Returns:
            Number of frames currently in buffer.
        """
        return self._frame_queue.qsize()

    def is_empty(self) -> bool:
        """Check if buffer is empty.

        Returns:
            True if buffer is empty, False otherwise.
        """
        return self._frame_queue.empty()

    def is_full(self) -> bool:
        """Check if buffer is full.

        Returns:
            True if buffer is full, False otherwise.
        """
        return self._frame_queue.full()

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
                self._audio_queue.get_nowait()
            except Empty:
                break
        self._frame_timestamps.clear()
        logger.debug("Buffer cleared")

    def stop(self) -> None:
        """Stop the buffer and signal consumers to stop."""
        self._stopped.set()
        # Unblock any waiting get() calls
        try:
            self._frame_queue.put_nowait((None, time.time()))
            self._audio_queue.put_nowait((None, time.time()))
        except Full:
            pass

    def get_current_source(self) -> Optional[str]:
        """Get the current source identifier.

        Returns:
            Current source identifier or None.
        """
        with self._source_lock:
            return self._current_source

    def get_statistics(self) -> dict:
        """Get buffer statistics.

        Returns:
            Dictionary with statistics:
            - frames_put: Total frames added
            - frames_get: Total frames retrieved
            - frames_dropped: Total frames dropped
            - current_size: Current buffer size
            - current_source: Current source identifier
        """
        return {
            "frames_put": self._frames_put,
            "frames_get": self._frames_get,
            "frames_dropped": self._frames_dropped,
            "current_size": self.size(),
            "current_source": self.get_current_source(),
        }


class AsyncFrameSource:
    """Adapter to feed async frame generators into the buffer."""

    def __init__(
        self,
        buffer: VideoFrameBuffer,
        source_name: str,
    ) -> None:
        """Initialize async frame source adapter.

        Args:
            buffer: Video frame buffer to feed frames into.
            source_name: Source identifier for tracking.
        """
        self.buffer = buffer
        self.source_name = source_name
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(
        self,
        frame_generator: AsyncIterator[np.ndarray],
        audio_generator: Optional[AsyncIterator[np.ndarray]] = None,
    ) -> None:
        """Start feeding frames from async generator into buffer.

        Args:
            frame_generator: Async iterator yielding video frames.
            audio_generator: Optional async iterator yielding audio chunks synchronized with frames.
        """
        if self._running:
            logger.warning(f"AsyncFrameSource {self.source_name} already running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._feed_loop(frame_generator, audio_generator)
        )

    async def stop(self) -> None:
        """Stop feeding frames."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _feed_loop(
        self,
        frame_generator: AsyncIterator[np.ndarray],
        audio_generator: Optional[AsyncIterator[np.ndarray]] = None,
    ) -> None:
        """Internal loop to feed frames from generator to buffer."""
        try:
            if audio_generator is not None:
                async for frame, audio in self._zip_generators(frame_generator, audio_generator):
                    if not self._running:
                        break
                    self.buffer.put_frame(frame, audio, source=self.source_name)
            else:
                async for frame in frame_generator:
                    if not self._running:
                        break
                    self.buffer.put_frame(frame, source=self.source_name)
        except asyncio.CancelledError:
            logger.debug(f"AsyncFrameSource {self.source_name} feed loop cancelled")
        except Exception as e:
            logger.error(f"Error in AsyncFrameSource {self.source_name} feed loop: {e}")
        finally:
            self._running = False

    async def _zip_generators(
        self,
        frame_gen: AsyncIterator[np.ndarray],
        audio_gen: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[Tuple[np.ndarray, np.ndarray]]:
        """Zip frame and audio generators together."""
        try:
            async for frame in frame_gen:
                try:
                    audio = await audio_gen.__anext__()
                    yield (frame, audio)
                except StopAsyncIteration:
                    # Audio generator ended, yield frame with None audio
                    yield (frame, None)
        except StopAsyncIteration:
            pass

