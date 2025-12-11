"""Base interface for Text-to-Speech (TTS) providers.

This module defines the abstract base class that all TTS providers must implement,
ensuring a consistent interface for text-to-speech synthesis.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class TTSError(Exception):
    """Base exception for TTS-related errors."""

    pass


class TTSProviderError(TTSError):
    """Error raised when TTS provider fails."""

    pass


class TTSFormatError(TTSError):
    """Error raised when audio format conversion fails."""

    pass


class TTSProvider(ABC):
    """Abstract base class for TTS providers.

    All TTS providers must implement this interface to ensure consistent
    audio output format and behavior. The standard audio format is:
    - Format: WAV
    - Sample rate: 44.1kHz (44100 Hz)
    - Channels: Mono (1 channel)
    """

    @abstractmethod
    async def synthesize(
        self, text: str, language: str = "en", voice_id: Optional[str] = None
    ) -> bytes:
        """Synthesize text to speech audio.

        Args:
            text: Text to convert to speech.
            language: Language code (e.g., 'en', 'es', 'fr'). Defaults to 'en'.
            voice_id: Optional voice ID for the provider. If None, provider
                will select a default voice for the language.

        Returns:
            Audio data as bytes in WAV format (44.1kHz, mono).

        Raises:
            TTSProviderError: If synthesis fails.
            TTSFormatError: If audio format conversion fails.
        """
        pass

    @abstractmethod
    async def synthesize_streaming(
        self, text: str, language: str = "en", voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Synthesize text to speech audio in streaming mode.

        This method yields audio chunks as they become available, enabling
        lower latency for long texts.

        Args:
            text: Text to convert to speech.
            language: Language code (e.g., 'en', 'es', 'fr'). Defaults to 'en'.
            voice_id: Optional voice ID for the provider. If None, provider
                will select a default voice for the language.

        Yields:
            Audio data chunks as bytes. Final audio will be in WAV format
            (44.1kHz, mono) when all chunks are concatenated.

        Raises:
            TTSProviderError: If synthesis fails.
            TTSFormatError: If audio format conversion fails.
        """
        pass

    @abstractmethod
    async def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available voices for the provider.

        Args:
            language: Optional language code to filter voices. If None,
                returns all available voices.

        Returns:
            List of voice dictionaries with at least 'id' and 'name' keys.

        Raises:
            TTSProviderError: If voice listing fails.
        """
        pass

    async def initialize(self) -> None:
        """Initialize the provider (load models, connect to API, etc.).

        This method should be called before using the provider to ensure
        all resources are ready. Default implementation does nothing.

        Raises:
            TTSProviderError: If initialization fails.
        """
        pass

    async def cleanup(self) -> None:
        """Clean up resources (unload models, close connections, etc.).

        This method should be called when the provider is no longer needed
        to free up resources. Default implementation does nothing.
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized and ready to use.

        Returns:
            True if provider is initialized, False otherwise.
            Default implementation returns True (assumes ready after __init__).
        """
        return True

