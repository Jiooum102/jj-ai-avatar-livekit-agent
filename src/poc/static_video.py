"""Static video generator for idle state streaming.

This module provides an async iterator that generates continuous static/default
video frames for the idle state, ensuring the RTMP stream never stops.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Optional

import cv2
import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)


class StaticVideoGenerator:
    """Async static video generator for continuous streaming.

    This generator yields frames at a consistent frame rate, supporting both
    static images (repeated frames) and video files (looped).

    Example:
        ```python
        async with StaticVideoGenerator() as generator:
            async for frame in generator:
                # Process frame (np.ndarray, RGB format)
                print(f"Frame shape: {frame.shape}")
        ```
    """

    def __init__(
        self,
        source_path: Optional[Path] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        fps: Optional[int] = None,
        loop: bool = True,
    ) -> None:
        """Initialize static video generator.

        Args:
            source_path: Path to static image or video file. If None, uses
                        settings from configuration.
            target_width: Target frame width. If None, uses RTMP resolution.
            target_height: Target frame height. If None, uses RTMP resolution.
            fps: Frame rate. If None, uses StaticVideoSettings.fps or RTMP fps.
            loop: Whether to loop video files. Defaults to True.
        """
        self.settings = get_settings()
        self.static_settings = self.settings.static_video
        self.rtmp_settings = self.settings.rtmp

        # Use provided values or fall back to settings
        self.source_path = source_path or self.static_settings.path
        self.target_width = target_width or self.rtmp_settings.width
        self.target_height = target_height or self.rtmp_settings.height
        self.loop = loop if source_path is None else self.static_settings.loop

        # Determine FPS: use provided, then static_settings, then RTMP default
        self.fps = fps or self.static_settings.fps or self.rtmp_settings.fps
        self.frame_interval = 1.0 / self.fps

        # Internal state
        self._source_type: Optional[str] = None  # 'image' or 'video'
        self._static_frame: Optional[np.ndarray] = None  # For static images
        self._video_cap: Optional[cv2.VideoCapture] = None  # For video files
        self._is_running = False
        self._last_frame_time: Optional[float] = None  # None until first frame

    async def __aenter__(self) -> "StaticVideoGenerator":
        """Async context manager entry."""
        await self._load_source()
        self._is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._is_running = False
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        logger.info("Static video generator closed")

    def __aiter__(self) -> "StaticVideoGenerator":
        """Async iterator protocol."""
        return self

    async def __anext__(self) -> np.ndarray:
        """Get next frame.

        Returns:
            np.ndarray: Frame as RGB numpy array with shape (height, width, 3).

        Raises:
            StopAsyncIteration: If generator is stopped.
        """
        if not self._is_running:
            raise StopAsyncIteration

        # Get frame based on source type
        frame = self._get_frame()

        if frame is None:
            # End of video (should not happen if looping, but handle gracefully)
            if self._source_type == "video" and self.loop:
                # Reset video to beginning
                if self._video_cap is not None:
                    self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame = self._get_frame()
            else:
                raise StopAsyncIteration

        # Resize frame to target resolution
        frame = self._resize_frame(frame, self.target_width, self.target_height)

        # Maintain frame rate by sleeping if needed
        current_time = asyncio.get_event_loop().time()
        if self._last_frame_time is not None:
            elapsed = current_time - self._last_frame_time
            if elapsed < self.frame_interval:
                await asyncio.sleep(self.frame_interval - elapsed)
        self._last_frame_time = asyncio.get_event_loop().time()

        return frame

    async def _load_source(self) -> None:
        """Load and validate the source file (image or video).

        Raises:
            FileNotFoundError: If source file does not exist.
            ValueError: If source file format is not supported.
        """
        source_path = Path(self.source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Determine if it's an image or video based on extension
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}

        ext = source_path.suffix.lower()

        if ext in image_extensions:
            self._source_type = "image"
            await self._load_image(source_path)
        elif ext in video_extensions:
            self._source_type = "video"
            await self._load_video(source_path)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported image formats: {image_extensions}, "
                f"Supported video formats: {video_extensions}"
            )

        logger.info(
            f"Loaded {self._source_type} source: {source_path} "
            f"(target resolution: {self.target_width}x{self.target_height}, fps: {self.fps})"
        )

    async def _load_image(self, image_path: Path) -> None:
        """Load static image file.

        Args:
            image_path: Path to image file.

        Raises:
            ValueError: If image cannot be loaded.
        """
        # Load image using OpenCV (BGR format)
        frame_bgr = cv2.imread(str(image_path))

        if frame_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Store the frame (will be repeated)
        self._static_frame = frame_rgb

        logger.debug(f"Loaded static image: {image_path} (shape: {frame_rgb.shape})")

    async def _load_video(self, video_path: Path) -> None:
        """Load video file.

        Args:
            video_path: Path to video file.

        Raises:
            ValueError: If video cannot be opened.
        """
        # Open video file
        self._video_cap = cv2.VideoCapture(str(video_path))

        if not self._video_cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Get video properties for logging
        video_fps = self._video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.debug(
            f"Opened video: {video_path} "
            f"(resolution: {video_width}x{video_height}, "
            f"fps: {video_fps}, frames: {frame_count}, loop: {self.loop})"
        )

    def _get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from source.

        Returns:
            np.ndarray: Frame in RGB format, or None if end of video (non-looping).
        """
        if self._source_type == "image":
            # Return the stored static frame
            return self._static_frame.copy() if self._static_frame is not None else None

        elif self._source_type == "video":
            if self._video_cap is None:
                return None

            # Read next frame from video
            ret, frame_bgr = self._video_cap.read()

            if not ret:
                # End of video
                return None

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb

        return None

    def _resize_frame(
        self, frame: np.ndarray, target_width: int, target_height: int
    ) -> np.ndarray:
        """Resize frame to target resolution.

        Args:
            frame: Input frame (RGB format).
            target_width: Target width in pixels.
            target_height: Target height in pixels.

        Returns:
            np.ndarray: Resized frame (RGB format).
        """
        current_height, current_width = frame.shape[:2]

        # Only resize if dimensions differ
        if current_width != target_width or current_height != target_height:
            frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )

        return frame

