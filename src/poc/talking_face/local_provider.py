"""Local talking face provider implementation.

This module provides local talking face generation using AI models like MuseTalk,
MimicTalk, and SyncTalk. It uses a strategy pattern to support multiple models.
"""

import logging
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import numpy as np

from src.config import get_settings
from src.poc.talking_face.base import (
    TalkingFaceModelError,
    TalkingFaceProvider,
    TalkingFaceProviderError,
)
from src.poc.talking_face.model_manager import ModelManager

logger = logging.getLogger(__name__)


class LocalTalkingFaceProvider(TalkingFaceProvider):
    """Local talking face provider using AI models.

    This provider uses local AI models (MuseTalk, MimicTalk, SyncTalk) for
    talking face generation. It supports model loading, inference, and
    resource management.
    """

    def __init__(
        self,
        model_type: str = "musetalk",
        checkpoint_path: Optional[Path] = None,
        avatar_image: Optional[Path] = None,
        avatar_video: Optional[Path] = None,
        device: str = "cuda",
        batch_size: int = 1,
        fps: Optional[int] = None,
        use_float16: bool = False,
        whisper_dir: Optional[str] = None,
        vae_type: str = "sd-vae",
        version: str = "v15",
        bbox_shift: int = 0,
        extra_margin: int = 10,
    ) -> None:
        """Initialize local talking face provider.

        Args:
            model_type: Model type ('musetalk', 'mimictalk', 'synctalk').
            checkpoint_path: Path to model checkpoints.
            avatar_image: Path to default avatar image.
            avatar_video: Optional path to avatar video for video-to-video.
            device: Device for inference ('cuda' or 'cpu'). Defaults to 'cuda'.
            batch_size: Inference batch size. Defaults to 1.
            fps: Output FPS. If None, uses default from settings.
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.avatar_image = avatar_image
        self.avatar_video = avatar_video
        self.device = device
        self.batch_size = batch_size
        self.fps = fps
        self.use_float16 = use_float16
        self.whisper_dir = whisper_dir
        self.vae_type = vae_type
        self.version = version
        self.bbox_shift = bbox_shift
        self.extra_margin = extra_margin
        self._model_manager: Optional[ModelManager] = None
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the local provider and load models."""
        try:
            # Initialize model manager
            self._model_manager = ModelManager(
                model_type=self.model_type,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                batch_size=self.batch_size,
                use_float16=self.use_float16,
                whisper_dir=self.whisper_dir,
                vae_type=self.vae_type,
                version=self.version,
                bbox_shift=self.bbox_shift,
                extra_margin=self.extra_margin,
            )

            # Load model
            await self._model_manager.load_model()
            self._model = self._model_manager.get_model()

            # Load default avatar if provided
            if self.avatar_image:
                # Resolve relative paths to absolute paths
                # Path.resolve() resolves relative to current working directory
                avatar_path = self.avatar_image.resolve()
                if not avatar_path.exists():
                    raise TalkingFaceProviderError(
                        f"Avatar image not found: {avatar_path}\n"
                        f"  Original path: {self.avatar_image}\n"
                        f"  Current working directory: {Path.cwd()}\n"
                        f"  Please check that the file exists or update MUSETALK_AVATAR_IMAGE in your .env file."
                    )
                # Update to resolved path
                self.avatar_image = avatar_path
                logger.info(f"Avatar image resolved to: {avatar_path}")

            self._initialized = True
            logger.info(f"Local talking face provider initialized with {self.model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize local provider: {e}")
            raise TalkingFaceProviderError(f"Failed to initialize local provider: {e}") from e

    async def cleanup(self) -> None:
        """Clean up local provider resources."""
        if self._model_manager:
            await self._model_manager.unload_model()
        self._model = None
        self._initialized = False
        logger.info("Local talking face provider cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    async def generate_from_audio(
        self,
        audio: bytes,
        avatar: Union[str, Path],
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
    ) -> AsyncIterator[np.ndarray]:
        """Generate talking face video from audio and avatar using local model.

        Args:
            audio: Audio data as bytes (WAV format).
            avatar: Path to avatar image file.
            fps: Target frame rate. Defaults to 30.
            resolution: Target resolution (width, height). Defaults to (1280, 720).

        Yields:
            Video frames as numpy arrays in RGB format.

        Raises:
            TalkingFaceProviderError: If generation fails.
            TalkingFaceModelError: If model inference fails.
        """
        if not self._initialized:
            raise TalkingFaceProviderError("Provider not initialized. Call initialize() first.")

        if not self._model:
            raise TalkingFaceModelError("Model not loaded")

        try:
            # Use model-specific implementation
            avatar_path = Path(avatar) if isinstance(avatar, str) else avatar
            target_fps = self.fps or fps

            # Delegate to model-specific generator
            async for frame in self._model.generate_from_audio(
                audio, avatar_path, target_fps, resolution
            ):
                yield frame

        except TalkingFaceModelError:
            raise
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
        """Generate talking face video from source video and audio using local model.

        Args:
            source_video: Path to source video file.
            audio: Audio data as bytes (WAV format).
            fps: Target frame rate. Defaults to 30.
            resolution: Target resolution (width, height). Defaults to (1280, 720).

        Yields:
            Video frames as numpy arrays in RGB format.

        Raises:
            TalkingFaceProviderError: If generation fails.
            TalkingFaceModelError: If model inference fails.
        """
        if not self._initialized:
            raise TalkingFaceProviderError("Provider not initialized. Call initialize() first.")

        if not self._model:
            raise TalkingFaceModelError("Model not loaded")

        try:
            source_path = Path(source_video) if isinstance(source_video, str) else source_video
            target_fps = self.fps or fps

            # Check if model supports video-to-video
            if not hasattr(self._model, "generate_video_to_video"):
                raise TalkingFaceModelError(
                    f"Model {self.model_type} does not support video-to-video generation"
                )

            # Delegate to model-specific generator
            async for frame in self._model.generate_video_to_video(
                source_path, audio, target_fps, resolution
            ):
                yield frame

        except TalkingFaceModelError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate talking face: {e}")
            raise TalkingFaceProviderError(f"Failed to generate talking face: {e}") from e

