"""Base interface for Talking Face providers.

This module defines the abstract base class that all talking face providers must
implement, ensuring a consistent interface for audio-to-video generation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Union

import numpy as np


class TalkingFaceError(Exception):
    """Base exception for talking face-related errors."""

    pass


class TalkingFaceProviderError(TalkingFaceError):
    """Error raised when talking face provider fails."""

    pass


class TalkingFaceModelError(TalkingFaceError):
    """Error raised when model loading or inference fails."""

    pass


class TalkingFaceProvider(ABC):
    """Abstract base class for talking face providers.

    All talking face providers must implement this interface to ensure consistent
    video output format and behavior. The standard video format is:
    - Format: numpy arrays (frames as np.ndarray)
    - Color space: RGB
    - Frame rate: Configurable (typically 30 FPS)
    - Resolution: Configurable (typically 1280x720)
    """

    @abstractmethod
    async def generate_from_audio(
        self,
        audio: bytes,
        avatar: Union[str, Path],
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
    ) -> AsyncIterator[np.ndarray]:
        """Generate talking face video from audio and avatar image/video.

        This is the primary method for generating talking face video. It takes
        audio data and an avatar (image or video) and generates synchronized
        talking face video frames.

        Args:
            audio: Audio data as bytes (typically WAV format from TTS).
            avatar: Path to avatar image or video file.
            fps: Target frame rate for output video. Defaults to 30.
            resolution: Target resolution as (width, height) tuple.
                Defaults to (1280, 720).

        Yields:
            Video frames as numpy arrays in RGB format with shape (H, W, 3).

        Raises:
            TalkingFaceProviderError: If generation fails.
            TalkingFaceModelError: If model inference fails.
        """
        pass

    @abstractmethod
    async def generate_video_to_video(
        self,
        source_video: Union[str, Path],
        audio: bytes,
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
    ) -> AsyncIterator[np.ndarray]:
        """Generate talking face video from source video and audio.

        This method supports video-to-video generation, where a source video
        is used as the base and audio drives the lip-sync. MuseTalk supports
        this capability.

        Args:
            source_video: Path to source video file.
            audio: Audio data as bytes (typically WAV format from TTS).
            fps: Target frame rate for output video. Defaults to 30.
            resolution: Target resolution as (width, height) tuple.
                Defaults to (1280, 720).

        Yields:
            Video frames as numpy arrays in RGB format with shape (H, W, 3).

        Raises:
            TalkingFaceProviderError: If generation fails.
            TalkingFaceModelError: If model inference fails.
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load models, connect to API, etc.).

        This method should be called before using the provider to ensure
        all resources are ready.

        Raises:
            TalkingFaceProviderError: If initialization fails.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (unload models, close connections, etc.).

        This method should be called when the provider is no longer needed
        to free up resources.
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if provider is initialized and ready to use.

        Returns:
            True if provider is initialized, False otherwise.
        """
        pass

