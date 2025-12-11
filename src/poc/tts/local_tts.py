"""Local TTS provider implementation using edge-tts.

This module provides a local TTS implementation using edge-tts (Windows Edge TTS),
which is free, high-quality, and requires no API key.
"""

import asyncio
import io
import logging
from typing import AsyncIterator, Optional

import edge_tts
from pydub import AudioSegment

from src.config import get_settings
from src.poc.tts.base import TTSFormatError, TTSProvider, TTSProviderError

logger = logging.getLogger(__name__)


class LocalTTSProvider(TTSProvider):
    """Local TTS provider using edge-tts.

    This provider uses edge-tts for text-to-speech synthesis. It supports
    multiple languages and voices, with automatic voice selection by language.
    """

    def __init__(
        self,
        provider: str = "edge-tts",
        voice: Optional[str] = None,
        language: str = "en",
        sample_rate: int = 44100,
        channels: int = 1,
    ) -> None:
        """Initialize local TTS provider.

        Args:
            provider: TTS provider type ('edge-tts' or 'pyttsx3'). Currently
                only 'edge-tts' is supported.
            voice: Optional voice ID (e.g., 'en-US-AriaNeural'). If None,
                voice will be auto-selected by language.
            language: Default language code (e.g., 'en', 'es', 'fr').
            sample_rate: Target audio sample rate (Hz). Defaults to 44100.
            channels: Target audio channels (1=mono, 2=stereo). Defaults to 1.
        """
        if provider != "edge-tts":
            raise ValueError(f"Unsupported local TTS provider: {provider}. Only 'edge-tts' is supported.")

        self.provider = provider
        self.voice = voice
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self._voice_cache: Optional[list[dict]] = None

    async def _get_voice(self, language: str, voice_id: Optional[str] = None) -> str:
        """Get voice ID for synthesis.

        Args:
            language: Language code.
            voice_id: Optional voice ID. If provided, returns as-is.

        Returns:
            Voice ID string.

        Raises:
            TTSProviderError: If voice selection fails.
        """
        if voice_id:
            return voice_id

        # Auto-select voice by language
        voices = await self.list_voices(language=language)
        if not voices:
            raise TTSProviderError(f"No voices available for language: {language}")

        # Prefer neural voices, fallback to first available
        neural_voices = [v for v in voices if "Neural" in v.get("name", "")]
        if neural_voices:
            return neural_voices[0]["id"]

        return voices[0]["id"]

    async def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available voices for edge-tts.

        Args:
            language: Optional language code to filter voices (e.g., 'en-US').

        Returns:
            List of voice dictionaries with 'id', 'name', 'gender', and 'locale' keys.
        """
        try:
            # Cache voices list to avoid repeated API calls
            if self._voice_cache is None:
                voices = await edge_tts.list_voices()
                self._voice_cache = [
                    {
                        "id": voice["ShortName"],
                        "name": voice["FriendlyName"],
                        "gender": voice["Gender"],
                        "locale": voice["Locale"],
                    }
                    for voice in voices
                ]

            voices_list = self._voice_cache

            # Filter by language if provided
            if language:
                # Normalize language code (e.g., 'en' -> 'en-US')
                language_prefix = language.lower()
                voices_list = [
                    v for v in voices_list if v["locale"].lower().startswith(language_prefix)
                ]

            return voices_list
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise TTSProviderError(f"Failed to list voices: {e}") from e

    def _convert_audio_format(self, audio_data: bytes) -> bytes:
        """Convert audio to standard format (WAV, 44.1kHz, mono).

        Args:
            audio_data: Input audio data (any format supported by pydub).

        Returns:
            Audio data in WAV format (44.1kHz, mono).

        Raises:
            TTSFormatError: If format conversion fails.
        """
        try:
            # Load audio from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_data))

            # Convert to mono if needed
            if audio.channels != self.channels:
                audio = audio.set_channels(self.channels)

            # Resample to target sample rate if needed
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)

            # Export to WAV format
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            return wav_buffer.getvalue()
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            raise TTSFormatError(f"Failed to convert audio format: {e}") from e

    async def synthesize(
        self, text: str, language: str = "en", voice_id: Optional[str] = None
    ) -> bytes:
        """Synthesize text to speech audio using edge-tts.

        Args:
            text: Text to convert to speech.
            language: Language code (e.g., 'en', 'es', 'fr'). Defaults to 'en'.
            voice_id: Optional voice ID. If None, auto-selects by language.

        Returns:
            Audio data as bytes in WAV format (44.1kHz, mono).

        Raises:
            TTSProviderError: If synthesis fails.
            TTSFormatError: If audio format conversion fails.
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Get voice ID
            voice = await self._get_voice(language, voice_id)

            # Synthesize using edge-tts
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # Convert to standard format
            return self._convert_audio_format(audio_data)
        except TTSFormatError:
            raise
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise TTSProviderError(f"Failed to synthesize speech: {e}") from e

    async def synthesize_streaming(
        self, text: str, language: str = "en", voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Synthesize text to speech audio in streaming mode.

        Args:
            text: Text to convert to speech.
            language: Language code (e.g., 'en', 'es', 'fr'). Defaults to 'en'.
            voice_id: Optional voice ID. If None, auto-selects by language.

        Yields:
            Audio data chunks as bytes. Final audio will be in WAV format
            (44.1kHz, mono) when all chunks are concatenated.

        Raises:
            TTSProviderError: If synthesis fails.
            TTSFormatError: If audio format conversion fails.
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Get voice ID
            voice = await self._get_voice(language, voice_id)

            # Stream synthesis using edge-tts
            communicate = edge_tts.Communicate(text, voice)
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            # Convert all chunks to standard format
            if audio_chunks:
                full_audio = b"".join(audio_chunks)
                converted_audio = self._convert_audio_format(full_audio)
                # Yield in reasonable chunks (e.g., 4KB)
                chunk_size = 4096
                for i in range(0, len(converted_audio), chunk_size):
                    yield converted_audio[i : i + chunk_size]
        except TTSFormatError:
            raise
        except Exception as e:
            logger.error(f"TTS streaming synthesis failed: {e}")
            raise TTSProviderError(f"Failed to synthesize speech: {e}") from e

