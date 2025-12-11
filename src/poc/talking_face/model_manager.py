"""Model manager for talking face models.

This module handles model loading, caching, and resource management for
different talking face models (MuseTalk, MimicTalk, SyncTalk).
"""

import logging
from pathlib import Path
from typing import Optional

from src.poc.talking_face.base import TalkingFaceModelError
from src.poc.talking_face.models.musetalk import MuseTalkModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for talking face model loading and management."""

    def __init__(
        self,
        model_type: str,
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda",
        batch_size: int = 1,
        use_float16: bool = False,
        whisper_dir: Optional[str] = None,
        vae_type: str = "sd-vae",
        version: str = "v15",
        bbox_shift: int = 0,
        extra_margin: int = 10,
    ) -> None:
        """Initialize model manager.

        Args:
            model_type: Model type ('musetalk', 'mimictalk', 'synctalk').
            checkpoint_path: Path to model checkpoints. If None, uses default.
            device: Device for inference ('cuda' or 'cpu'). Defaults to 'cuda'.
            batch_size: Inference batch size. Defaults to 1.
            use_float16: Use float16 for faster inference. Defaults to False.
            whisper_dir: Path to Whisper model directory. Defaults to None.
            vae_type: VAE type. Defaults to 'sd-vae'.
            version: Model version ('v1' or 'v15'). Defaults to 'v15'.
            bbox_shift: Bounding box shift value. Defaults to 0.
            extra_margin: Extra margin for face cropping. Defaults to 10.
        """
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.batch_size = batch_size
        self.use_float16 = use_float16
        self.whisper_dir = whisper_dir
        self.vae_type = vae_type
        self.version = version
        self.bbox_shift = bbox_shift
        self.extra_margin = extra_margin
        self._model: Optional[MuseTalkModel] = None

    async def load_model(self) -> None:
        """Load the specified model.

        Raises:
            TalkingFaceModelError: If model loading fails.
        """
        try:
            if self.model_type == "musetalk":
                if self.checkpoint_path is None:
                    # Use default path
                    self.checkpoint_path = Path("./models/musetalk")

                self._model = MuseTalkModel(
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
                await self._model.load()
            elif self.model_type == "mimictalk":
                # TODO: Implement MimicTalk model loading
                raise TalkingFaceModelError("MimicTalk model not yet implemented")
            elif self.model_type == "synctalk":
                # TODO: Implement SyncTalk model loading
                raise TalkingFaceModelError("SyncTalk model not yet implemented")
            else:
                raise TalkingFaceModelError(f"Unknown model type: {self.model_type}")

            logger.info(f"Model {self.model_type} loaded successfully")
        except TalkingFaceModelError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_type}: {e}")
            raise TalkingFaceModelError(f"Failed to load model: {e}") from e

    async def unload_model(self) -> None:
        """Unload the current model from memory."""
        if self._model:
            if hasattr(self._model, "unload"):
                await self._model.unload()
            self._model = None
            logger.info(f"Model {self.model_type} unloaded")

    def get_model(self):
        """Get the loaded model instance.

        Returns:
            Loaded model instance.

        Raises:
            TalkingFaceModelError: If model is not loaded.
        """
        if self._model is None:
            raise TalkingFaceModelError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

