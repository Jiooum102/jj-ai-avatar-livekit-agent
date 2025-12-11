"""API-based TTS provider implementation.

This module provides TTS implementations using external API services such as
ElevenLabs, OpenAI, and Cartesia.
"""

import asyncio
import io
import logging
from typing import AsyncIterator, Optional

import httpx
from pydub import AudioSegment

from src.poc.tts.base import TTSFormatError, TTSProvider, TTSProviderError

logger = logging.getLogger(__name__)


class APITTSProvider(TTSProvider):
    """API-based TTS provider.

    This provider supports multiple TTS API services including ElevenLabs,
    OpenAI, and Cartesia. It handles authentication, retries, and audio
    format conversion.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        voice_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        sample_rate: int = 44100,
        channels: int = 1,
    ) -> None:
        """Initialize API TTS provider.

        Args:
            provider: API provider name ('elevenlabs', 'openai', or 'cartesia').
            api_key: API key for authentication.
            voice_id: Optional voice ID for the provider.
            endpoint: Optional custom API endpoint URL.
            timeout: Request timeout in seconds. Defaults to 30.
            max_retries: Maximum number of retry attempts. Defaults to 3.
            sample_rate: Target audio sample rate (Hz). Defaults to 44100.
            channels: Target audio channels (1=mono, 2=stereo). Defaults to 1.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.voice_id = voice_id
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.sample_rate = sample_rate
        self.channels = channels

        # Provider-specific endpoints
        self._endpoints = {
            "elevenlabs": endpoint or "https://api.elevenlabs.io/v1/text-to-speech",
            "openai": endpoint or "https://api.openai.com/v1/audio/speech",
            "cartesia": endpoint or "https://api.cartesia.ai/v1/tts",
        }

        if self.provider not in self._endpoints:
            raise ValueError(
                f"Unsupported API provider: {provider}. "
                f"Supported providers: {list(self._endpoints.keys())}"
            )

    def _get_endpoint(self, voice_id: Optional[str] = None) -> str:
        """Get API endpoint URL for synthesis.

        Args:
            voice_id: Optional voice ID to include in URL.

        Returns:
            API endpoint URL.
        """
        base_url = self._endpoints[self.provider]

        if self.provider == "elevenlabs":
            # ElevenLabs: /v1/text-to-speech/{voice_id}
            if voice_id or self.voice_id:
                voice = voice_id or self.voice_id
                return f"{base_url}/{voice}"
            return base_url
        elif self.provider == "openai":
            # OpenAI: /v1/audio/speech
            return base_url
        elif self.provider == "cartesia":
            # Cartesia: /v1/tts
            return base_url

        return base_url

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API request.

        Returns:
            Dictionary of HTTP headers.
        """
        headers = {
            "Content-Type": "application/json",
        }

        if self.provider == "elevenlabs":
            headers["xi-api-key"] = self.api_key
        elif self.provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "cartesia":
            headers["x-api-key"] = self.api_key

        return headers

    def _get_request_payload(self, text: str, voice_id: Optional[str] = None) -> dict:
        """Get request payload for API call.

        Args:
            text: Text to synthesize.
            voice_id: Optional voice ID.

        Returns:
            Request payload dictionary.
        """
        voice = voice_id or self.voice_id

        if self.provider == "elevenlabs":
            return {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                },
            }
        elif self.provider == "openai":
            return {
                "model": "tts-1",
                "input": text,
                "voice": voice or "alloy",
            }
        elif self.provider == "cartesia":
            return {
                "text": text,
                "voice_id": voice,
            }

        return {"text": text}

    def _convert_audio_format(self, audio_data: bytes, format_hint: str = "mp3") -> bytes:
        """Convert audio to standard format (WAV, 44.1kHz, mono).

        Args:
            audio_data: Input audio data.
            format_hint: Hint for audio format (e.g., 'mp3', 'wav').

        Returns:
            Audio data in WAV format (44.1kHz, mono).

        Raises:
            TTSFormatError: If format conversion fails.
        """
        try:
            # Load audio from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format_hint)

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

    async def _make_request(
        self, text: str, voice_id: Optional[str] = None, streaming: bool = False
    ) -> bytes:
        """Make API request with retry logic.

        Args:
            text: Text to synthesize.
            voice_id: Optional voice ID.
            streaming: Whether to return streaming response.

        Returns:
            Audio data as bytes or async iterator of bytes.

        Raises:
            TTSProviderError: If request fails after retries.
        """
        endpoint = self._get_endpoint(voice_id)
        headers = self._get_headers()
        payload = self._get_request_payload(text, voice_id)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if streaming and self.provider == "elevenlabs":
                        # ElevenLabs supports streaming
                        async with client.stream(
                            "POST", endpoint, headers=headers, json=payload
                        ) as response:
                            response.raise_for_status()
                            audio_chunks = []
                            async for chunk in response.aiter_bytes():
                                audio_chunks.append(chunk)
                            return b"".join(audio_chunks)
                    else:
                        # Non-streaming request
                        response = await client.post(endpoint, headers=headers, json=payload)
                        response.raise_for_status()
                        return response.content
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
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
                    # Client error, don't retry
                    raise TTSProviderError(f"API request failed: {e}") from e
            except httpx.RequestError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Request error, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                last_error = e
            except Exception as e:
                raise TTSProviderError(f"Unexpected error: {e}") from e

        # All retries exhausted
        raise TTSProviderError(f"API request failed after {self.max_retries} attempts: {last_error}") from last_error

    async def synthesize(
        self, text: str, language: str = "en", voice_id: Optional[str] = None
    ) -> bytes:
        """Synthesize text to speech audio using API.

        Args:
            text: Text to convert to speech.
            language: Language code (ignored for most API providers).
            voice_id: Optional voice ID. If None, uses provider default.

        Returns:
            Audio data as bytes in WAV format (44.1kHz, mono).

        Raises:
            TTSProviderError: If synthesis fails.
            TTSFormatError: If audio format conversion fails.
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Make API request
            audio_data = await self._make_request(text, voice_id, streaming=False)
            # _make_request returns bytes, not AsyncIterator for non-streaming

            # Convert to standard format
            # Most APIs return MP3, but check format
            format_hint = "mp3" if self.provider in ["elevenlabs", "openai"] else "wav"
            return self._convert_audio_format(audio_data, format_hint=format_hint)
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
            language: Language code (ignored for most API providers).
            voice_id: Optional voice ID. If None, uses provider default.

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
            # Make API request (streaming if supported)
            audio_data = await self._make_request(text, voice_id, streaming=True)
            # _make_request returns bytes for streaming (collected chunks)
            # Convert to standard format and yield in chunks
            format_hint = "mp3" if self.provider in ["elevenlabs", "openai"] else "wav"
            converted_audio = self._convert_audio_format(audio_data, format_hint=format_hint)
            chunk_size = 4096
            for i in range(0, len(converted_audio), chunk_size):
                yield converted_audio[i : i + chunk_size]
        except TTSFormatError:
            raise
        except Exception as e:
            logger.error(f"TTS streaming synthesis failed: {e}")
            raise TTSProviderError(f"Failed to synthesize speech: {e}") from e

    async def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available voices for the API provider.

        Args:
            language: Optional language code to filter voices (not all providers support this).

        Returns:
            List of voice dictionaries with at least 'id' and 'name' keys.

        Raises:
            TTSProviderError: If voice listing fails.
        """
        try:
            endpoint = self._endpoints[self.provider]
            headers = self._get_headers()

            if self.provider == "elevenlabs":
                # ElevenLabs: GET /v1/voices
                voices_endpoint = endpoint.replace("/text-to-speech", "/voices")
            elif self.provider == "openai":
                # OpenAI doesn't have a voices endpoint, return default voices
                return [
                    {"id": "alloy", "name": "Alloy"},
                    {"id": "echo", "name": "Echo"},
                    {"id": "fable", "name": "Fable"},
                    {"id": "onyx", "name": "Onyx"},
                    {"id": "nova", "name": "Nova"},
                    {"id": "shimmer", "name": "Shimmer"},
                ]
            elif self.provider == "cartesia":
                # Cartesia: GET /v1/voices
                voices_endpoint = f"{endpoint}/voices"
            else:
                return []

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(voices_endpoint, headers=headers)
                response.raise_for_status()
                data = response.json()

                if self.provider == "elevenlabs":
                    voices = [
                        {
                            "id": voice["voice_id"],
                            "name": voice.get("name", voice["voice_id"]),
                            "category": voice.get("category", ""),
                        }
                        for voice in data.get("voices", [])
                    ]
                elif self.provider == "cartesia":
                    voices = [
                        {
                            "id": voice.get("id", ""),
                            "name": voice.get("name", ""),
                        }
                        for voice in data.get("voices", [])
                    ]
                else:
                    voices = []

                # Filter by language if provided
                if language and voices:
                    # Most APIs don't support language filtering, return all
                    pass

                return voices
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise TTSProviderError(f"Failed to list voices: {e}") from e

