"""API-based talking face provider implementation.

This module provides talking face generation using external API services such as
Hedra, Tavus, D-ID, and similar services.
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import cv2
import httpx
import numpy as np

from src.poc.talking_face.base import (
    TalkingFaceModelError,
    TalkingFaceProvider,
    TalkingFaceProviderError,
)

logger = logging.getLogger(__name__)


class APITalkingFaceProvider(TalkingFaceProvider):
    """API-based talking face provider.

    This provider uses external APIs for talking face generation. It supports
    multiple API services and handles authentication, retries, and video
    format conversion.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        avatar_id: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """Initialize API talking face provider.

        Args:
            url: API endpoint URL.
            api_key: API key for authentication.
            avatar_id: Optional avatar ID for the API.
            timeout: Request timeout in seconds. Defaults to 30.
            max_retries: Maximum number of retry attempts. Defaults to 3.
        """
        self.url = url
        self.api_key = api_key
        self.avatar_id = avatar_id
        self.timeout = timeout
        self.max_retries = max_retries
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the API provider.

        For API providers, initialization typically involves validating
        credentials and checking API availability.
        """
        try:
            # Validate API connection
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                # Simple health check endpoint (adjust based on API)
                response = await client.get(f"{self.url}/health", headers=headers)
                response.raise_for_status()
            self._initialized = True
            logger.info("API talking face provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize API provider: {e}")
            raise TalkingFaceProviderError(f"Failed to initialize API provider: {e}") from e

    async def cleanup(self) -> None:
        """Clean up API provider resources."""
        self._initialized = False
        logger.info("API talking face provider cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    async def _make_request(
        self, audio: bytes, avatar: Union[str, Path], is_video: bool = False
    ) -> bytes:
        """Make API request with retry logic.

        Args:
            audio: Audio data as bytes.
            avatar: Path to avatar file or avatar ID.
            is_video: Whether avatar is a video file.

        Returns:
            Generated video data as bytes.

        Raises:
            TalkingFaceProviderError: If request fails after retries.
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    # Prepare request data
                    files = {"audio": ("audio.wav", audio, "audio/wav")}

                    if is_video:
                        files["video"] = (
                            "video.mp4",
                            open(avatar, "rb").read() if isinstance(avatar, (str, Path)) else avatar,
                            "video/mp4",
                        )
                    else:
                        files["image"] = (
                            "image.png",
                            open(avatar, "rb").read() if isinstance(avatar, (str, Path)) else avatar,
                            "image/png",
                        )

                    data = {}
                    if self.avatar_id:
                        data["avatar_id"] = self.avatar_id

                    response = await client.post(
                        f"{self.url}/generate", headers=headers, files=files, data=data
                    )
                    response.raise_for_status()
                    return response.content
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    last_error = e
                    continue
                elif e.response.status_code >= 500:  # Server error
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    last_error = e
                    continue
                else:
                    raise TalkingFaceProviderError(f"API request failed: {e}") from e
            except httpx.RequestError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Request error, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                last_error = e
            except Exception as e:
                raise TalkingFaceProviderError(f"Unexpected error: {e}") from e

        raise TalkingFaceProviderError(
            f"API request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _extract_frames_from_video(
        self, video_data: bytes, fps: int, resolution: tuple[int, int]
    ) -> list[np.ndarray]:
        """Extract frames from video data.

        Args:
            video_data: Video data as bytes.
            fps: Target frame rate.
            resolution: Target resolution (width, height).

        Returns:
            List of video frames as numpy arrays.
        """
        # Write video data to temporary buffer
        video_buffer = io.BytesIO(video_data)
        video_buffer.seek(0)

        # Use OpenCV to read video
        cap = cv2.VideoCapture()
        # Note: cv2.VideoCapture doesn't directly support BytesIO, so we'd need
        # to write to a temporary file or use a different approach
        # For now, this is a placeholder that would need actual implementation
        frames = []
        # TODO: Implement proper video frame extraction from bytes
        return frames

    async def generate_from_audio(
        self,
        audio: bytes,
        avatar: Union[str, Path],
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
    ) -> AsyncIterator[np.ndarray]:
        """Generate talking face video from audio and avatar using API.

        Args:
            audio: Audio data as bytes (WAV format).
            avatar: Path to avatar image file.
            fps: Target frame rate. Defaults to 30.
            resolution: Target resolution (width, height). Defaults to (1280, 720).

        Yields:
            Video frames as numpy arrays in RGB format.

        Raises:
            TalkingFaceProviderError: If generation fails.
        """
        if not self._initialized:
            raise TalkingFaceProviderError("Provider not initialized. Call initialize() first.")

        try:
            # Check if avatar is image or video
            avatar_path = Path(avatar) if isinstance(avatar, str) else avatar
            is_video = avatar_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]

            # Make API request
            video_data = await self._make_request(audio, avatar, is_video=is_video)

            # Extract frames from video
            frames = self._extract_frames_from_video(video_data, fps, resolution)

            # Yield frames
            for frame in frames:
                # Resize if needed
                if frame.shape[:2] != (resolution[1], resolution[0]):
                    frame = cv2.resize(frame, resolution)
                # Convert BGR to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame

        except Exception as e:
            logger.error(f"Failed to generate talking face: {e}")
            raise TalkingFaceProviderError(f"Failed to generate talking face: {e}") from e

    async def generate_video_to_video(
        self,
        source_video: Union[str, Path],
        audio: bytes,
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
    ) -> AsyncIterator[np.ndarray]:
        """Generate talking face video from source video and audio using API.

        Args:
            source_video: Path to source video file.
            audio: Audio data as bytes (WAV format).
            fps: Target frame rate. Defaults to 30.
            resolution: Target resolution (width, height). Defaults to (1280, 720).

        Yields:
            Video frames as numpy arrays in RGB format.

        Raises:
            TalkingFaceProviderError: If generation fails.
        """
        # For API providers, video-to-video is similar to image-to-video
        # but with video input
        async for frame in self.generate_from_audio(audio, source_video, fps, resolution):
            yield frame

