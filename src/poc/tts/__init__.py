"""Text-to-Speech (TTS) module.

This module provides text-to-speech synthesis capabilities with support for
both local (edge-tts) and API-based providers (ElevenLabs, OpenAI, Cartesia).

The module ensures all providers return audio in a consistent format:
- Format: WAV
- Sample rate: 44.1kHz (44100 Hz)
- Channels: Mono (1 channel)
"""

from src.poc.tts.api_tts import APITTSProvider
from src.poc.tts.base import (
    TTSError,
    TTSFormatError,
    TTSProvider,
    TTSProviderError,
)
from src.poc.tts.factory import create_tts_provider, health_check
from src.poc.tts.local_tts import LocalTTSProvider

__all__ = [
    "TTSProvider",
    "LocalTTSProvider",
    "APITTSProvider",
    "create_tts_provider",
    "health_check",
    "TTSError",
    "TTSProviderError",
    "TTSFormatError",
]

