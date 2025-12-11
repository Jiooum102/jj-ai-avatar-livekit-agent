"""Factory for creating talking face provider instances.

This module provides a factory function to create and configure talking face
providers based on application settings.
"""

import logging
from typing import Optional

from src.config import get_settings
from src.config.settings import TalkingFaceSettings
from src.poc.talking_face.api_provider import APITalkingFaceProvider
from src.poc.talking_face.base import TalkingFaceProvider, TalkingFaceProviderError
from src.poc.talking_face.local_provider import LocalTalkingFaceProvider

logger = logging.getLogger(__name__)


def create_talking_face_provider(
    settings: Optional[TalkingFaceSettings] = None,
) -> TalkingFaceProvider:
    """Create and configure a talking face provider based on settings.

    Args:
        settings: Optional talking face settings. If None, uses settings from
            get_settings().talking_face.

    Returns:
        Configured talking face provider instance.

    Raises:
        TalkingFaceProviderError: If provider creation fails or required settings
            are missing.
    """
    if settings is None:
        settings = get_settings().talking_face

    provider_type = settings.provider

    if provider_type == "api":
        # Create API provider
        api_settings = settings.api
        if not api_settings.api_key:
            raise TalkingFaceProviderError(
                "API talking face provider requires TALKING_FACE_API_API_KEY to be set."
            )

        if not api_settings.url:
            raise TalkingFaceProviderError(
                "API talking face provider requires TALKING_FACE_API_URL to be set."
            )

        return APITalkingFaceProvider(
            url=api_settings.url,
            api_key=api_settings.api_key,
            avatar_id=api_settings.avatar_id,
            timeout=api_settings.timeout,
            max_retries=api_settings.max_retries,
        )
    elif provider_type == "local":
        # Create local provider
        model_settings = None
        if settings.model == "musetalk":
            model_settings = settings.musetalk
        elif settings.model == "mimictalk":
            model_settings = settings.mimictalk
        elif settings.model == "synctalk":
            model_settings = settings.synctalk
        else:
            raise TalkingFaceProviderError(f"Unknown model type: {settings.model}")

        if model_settings is None:
            raise TalkingFaceProviderError(f"Settings for model {settings.model} not found")

        return LocalTalkingFaceProvider(
            model_type=settings.model,
            checkpoint_path=model_settings.checkpoint_path,
            avatar_image=model_settings.avatar_image,
            avatar_video=getattr(model_settings, "avatar_video", None),
            device=model_settings.device,
            batch_size=model_settings.batch_size,
            fps=getattr(model_settings, "fps", None),
            use_float16=getattr(model_settings, "use_float16", False),
            whisper_dir=getattr(model_settings, "whisper_dir", None),
            vae_type=getattr(model_settings, "vae_type", "sd-vae"),
            version=getattr(model_settings, "version", "v15"),
            bbox_shift=getattr(model_settings, "bbox_shift", 0),
            extra_margin=getattr(model_settings, "extra_margin", 10),
        )
    else:
        raise TalkingFaceProviderError(f"Unknown provider type: {provider_type}")


async def health_check(provider: TalkingFaceProvider) -> bool:
    """Check if talking face provider is healthy and ready to use.

    Args:
        provider: Talking face provider instance to check.

    Returns:
        True if provider is healthy, False otherwise.
    """
    try:
        # Check if provider is initialized
        if not provider.is_initialized:
            return False

        # For local providers, check if model is loaded
        if isinstance(provider, LocalTalkingFaceProvider):
            # Check if model manager has loaded model
            if hasattr(provider, "_model_manager") and provider._model_manager:
                return provider._model_manager.is_loaded

        return True
    except Exception as e:
        logger.warning(f"Talking face provider health check failed: {e}")
        return False

