"""Factory for creating TTS provider instances.

This module provides a factory function to create and configure TTS providers
based on application settings.
"""

import logging
from typing import Optional

from src.config import get_settings
from src.config.settings import TTSSettings
from src.poc.tts.api_tts import APITTSProvider
from src.poc.tts.base import TTSProvider, TTSProviderError
from src.poc.tts.local_tts import LocalTTSProvider

logger = logging.getLogger(__name__)


def create_tts_provider(settings: Optional[TTSSettings] = None) -> TTSProvider:
    """Create and configure a TTS provider based on settings.

    Args:
        settings: Optional TTS settings. If None, uses settings from
            get_settings().tts.

    Returns:
        Configured TTS provider instance.

    Raises:
        TTSProviderError: If provider creation fails or required settings
            are missing.
    """
    if settings is None:
        settings = get_settings().tts

    provider_type = settings.provider

    if provider_type == "local":
        # Create local TTS provider
        local_settings = settings.local
        return LocalTTSProvider(
            provider=local_settings.provider,
            voice=local_settings.voice,
            language=local_settings.language,
            sample_rate=local_settings.sample_rate,
            channels=local_settings.channels,
        )
    elif provider_type == "api":
        # Create API TTS provider
        api_settings = settings.api
        if api_settings is None:
            raise TTSProviderError(
                "API TTS provider requires api settings. "
                "Set TTS_API_PROVIDER, TTS_API_KEY, and other TTS_API_* environment variables."
            )

        if not api_settings.api_key:
            raise TTSProviderError("API TTS provider requires TTS_API_KEY to be set.")

        return APITTSProvider(
            provider=api_settings.provider,
            api_key=api_settings.api_key,
            voice_id=api_settings.voice_id,
            endpoint=api_settings.endpoint,
            timeout=api_settings.timeout,
            max_retries=api_settings.max_retries,
            sample_rate=44100,  # Standard format
            channels=1,  # Standard format (mono)
        )
    else:
        raise TTSProviderError(f"Unknown TTS provider type: {provider_type}")


async def health_check(provider: TTSProvider) -> bool:
    """Check if TTS provider is healthy and ready to use.

    Args:
        provider: TTS provider instance to check.

    Returns:
        True if provider is healthy, False otherwise.
    """
    try:
        # Try to synthesize a short test phrase
        test_audio = await provider.synthesize("test", language="en")
        return len(test_audio) > 0
    except Exception as e:
        logger.warning(f"TTS provider health check failed: {e}")
        return False

