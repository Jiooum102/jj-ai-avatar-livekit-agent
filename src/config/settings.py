"""Configuration settings for the PoC application.

This module provides type-safe configuration management using Pydantic,
with support for environment variables and default values.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RabbitMQSettings(BaseSettings):
    """RabbitMQ connection settings."""

    model_config = SettingsConfigDict(env_prefix="RABBITMQ_", case_sensitive=False)

    host: str = Field(default="localhost", description="RabbitMQ server host")
    port: int = Field(default=5672, description="RabbitMQ server port")
    user: str = Field(default="guest", description="RabbitMQ username")
    password: str = Field(default="guest", description="RabbitMQ password")
    vhost: str = Field(default="/", description="RabbitMQ virtual host")
    queue: str = Field(default="talking_face_input", description="Queue name to consume from")
    exchange: Optional[str] = Field(default=None, description="Exchange name (optional)")
    routing_key: Optional[str] = Field(default=None, description="Routing key (optional)")

    @property
    def connection_url(self) -> str:
        """Get RabbitMQ connection URL."""
        return f"amqp://{self.user}:{self.password}@{self.host}:{self.port}/{self.vhost}"


class RTMPSettings(BaseSettings):
    """RTMP streaming settings."""

    model_config = SettingsConfigDict(env_prefix="RTMP_", case_sensitive=False)

    url: str = Field(..., description="RTMP output URL (required)")
    resolution: str = Field(default="1280x720", description="Video resolution (WIDTHxHEIGHT)")
    fps: int = Field(default=30, ge=1, le=60, description="Frame rate (frames per second)")
    bitrate: str = Field(default="2000k", description="Video bitrate")
    audio_bitrate: str = Field(default="128k", description="Audio bitrate")
    audio_sample_rate: int = Field(default=44100, description="Audio sample rate (Hz)")

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        """Validate resolution format."""
        parts = v.split("x")
        if len(parts) != 2:
            raise ValueError("Resolution must be in format WIDTHxHEIGHT (e.g., 1280x720)")
        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            raise ValueError("Resolution width and height must be integers")
        return v

    @property
    def width(self) -> int:
        """Get video width from resolution."""
        return int(self.resolution.split("x")[0])

    @property
    def height(self) -> int:
        """Get video height from resolution."""
        return int(self.resolution.split("x")[1])


class StaticVideoSettings(BaseSettings):
    """Static/default video settings for idle state."""

    model_config = SettingsConfigDict(env_prefix="STATIC_VIDEO_", case_sensitive=False)

    path: Path = Field(default=Path("./assets/default_avatar.png"), description="Path to static image or video")
    loop: bool = Field(default=True, description="Loop static video if it's a video file")
    fps: Optional[int] = Field(default=None, description="Override FPS for static video (uses RTMP FPS if None)")

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v


class LocalTTSSettings(BaseSettings):
    """Local TTS provider settings."""

    model_config = SettingsConfigDict(env_prefix="TTS_LOCAL_", case_sensitive=False)

    provider: Literal["edge-tts", "pyttsx3"] = Field(
        default="edge-tts", description="Local TTS provider: 'edge-tts' (recommended) or 'pyttsx3'"
    )
    voice: Optional[str] = Field(
        default=None, description="Voice ID (e.g., 'en-US-AriaNeural' for edge-tts, auto-selects if None)"
    )
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es', 'fr')")
    sample_rate: int = Field(default=44100, description="Audio sample rate (Hz)")
    channels: int = Field(default=1, ge=1, le=2, description="Audio channels (1=mono, 2=stereo)")


class APITTSSettings(BaseSettings):
    """API-based TTS provider settings."""

    model_config = SettingsConfigDict(env_prefix="TTS_API_", case_sensitive=False)

    provider: Literal["elevenlabs", "openai", "cartesia"] = Field(
        ..., description="API TTS provider: 'elevenlabs', 'openai', or 'cartesia'"
    )
    api_key: str = Field(..., description="API key for authentication")
    voice_id: Optional[str] = Field(default=None, description="Voice ID for the API provider")
    endpoint: Optional[str] = Field(default=None, description="Custom API endpoint URL (optional)")
    timeout: int = Field(default=30, description="API request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retry attempts")


class TTSSettings(BaseSettings):
    """Text-to-Speech settings."""

    model_config = SettingsConfigDict(env_prefix="TTS_", case_sensitive=False)

    provider: Literal["local", "api"] = Field(
        default="local", description="TTS provider type: 'local' or 'api'"
    )

    # Nested settings
    local: LocalTTSSettings = Field(default_factory=LocalTTSSettings)
    api: Optional[APITTSSettings] = Field(default=None, description="API settings (required if provider='api')")


class TalkingFaceAPISettings(BaseSettings):
    """Talking face API settings (for external API provider)."""

    model_config = SettingsConfigDict(env_prefix="TALKING_FACE_API_", case_sensitive=False)

    url: Optional[str] = Field(default=None, description="API endpoint URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    avatar_id: Optional[str] = Field(default=None, description="Avatar ID for API")
    timeout: int = Field(default=30, description="API request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")


class MuseTalkSettings(BaseSettings):
    """MuseTalk model-specific settings."""

    model_config = SettingsConfigDict(env_prefix="MUSETALK_", case_sensitive=False)

    checkpoint_path: Path = Field(
        default=Path("./models/musetalk"), description="Path to MuseTalk model checkpoints"
    )
    avatar_image: Path = Field(
        default=Path("./assets/avatar.png"), description="Default avatar image path"
    )
    avatar_video: Optional[Path] = Field(
        default=None, description="Optional: video input for MuseTalk video-to-video"
    )
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device for inference (cuda or cpu)")
    batch_size: int = Field(default=1, ge=1, description="Inference batch size")
    fps: Optional[int] = Field(default=None, description="Output FPS (uses RTMP FPS if None)")

    @field_validator("checkpoint_path", "avatar_image", "avatar_video", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path | None) -> Path | None:
        """Convert string to Path."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class MimicTalkSettings(BaseSettings):
    """MimicTalk model-specific settings (backup model for post-PoC evaluation)."""

    model_config = SettingsConfigDict(env_prefix="MIMICTALK_", case_sensitive=False)

    checkpoint_path: Path = Field(
        default=Path("./models/mimictalk"), description="Path to MimicTalk model checkpoints"
    )
    avatar_image: Path = Field(
        default=Path("./assets/avatar.png"), description="Identity image for MimicTalk"
    )
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device for inference (cuda or cpu)")
    batch_size: int = Field(default=1, ge=1, description="Inference batch size")


class SyncTalkSettings(BaseSettings):
    """SyncTalk model-specific settings (backup model for post-PoC evaluation)."""

    model_config = SettingsConfigDict(env_prefix="SYNCTALK_", case_sensitive=False)

    checkpoint_path: Path = Field(
        default=Path("./models/synctalk"), description="Path to SyncTalk model checkpoints"
    )
    workspace_path: Path = Field(
        default=Path("./models/synctalk/workspace"), description="SyncTalk workspace path"
    )
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device for inference (cuda or cpu)")
    batch_size: int = Field(default=1, ge=1, description="Inference batch size")


class TalkingFaceSettings(BaseSettings):
    """Talking face provider settings."""

    model_config = SettingsConfigDict(env_prefix="TALKING_FACE_", case_sensitive=False)

    provider: Literal["api", "local"] = Field(
        default="local", description="Provider type: 'api' for external API, 'local' for local AI inference"
    )
    model: Literal["musetalk", "mimictalk", "synctalk"] = Field(
        default="musetalk", description="Model type (primary: musetalk, backup: mimictalk, synctalk)"
    )

    # Nested settings
    api: TalkingFaceAPISettings = Field(default_factory=TalkingFaceAPISettings)
    musetalk: MuseTalkSettings = Field(default_factory=MuseTalkSettings)
    mimictalk: MimicTalkSettings = Field(default_factory=MimicTalkSettings)
    synctalk: SyncTalkSettings = Field(default_factory=SyncTalkSettings)


class FFmpegSettings(BaseSettings):
    """FFmpeg configuration settings."""

    model_config = SettingsConfigDict(env_prefix="FFMPEG_", case_sensitive=False)

    path: str = Field(default="ffmpeg", description="FFmpeg executable path (or 'ffmpeg' if in PATH)")
    preset: str = Field(default="ultrafast", description="Encoding preset (ultrafast, fast, medium, slow)")
    tune: str = Field(default="zerolatency", description="Encoding tune (zerolatency for low latency)")
    gop_size: int = Field(default=50, description="GOP (Group of Pictures) size")
    pixel_format: str = Field(default="yuv420p", description="Output pixel format")


class Settings(BaseSettings):
    """Main application settings.

    This class aggregates all configuration settings and provides
    a single entry point for application configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="jj-ai-avatar-livekit-agent-poc", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Component settings
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    rtmp: RTMPSettings = Field(
        default_factory=lambda: RTMPSettings()
    )  # Will read from RTMP_URL env var
    static_video: StaticVideoSettings = Field(default_factory=StaticVideoSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    talking_face: TalkingFaceSettings = Field(default_factory=TalkingFaceSettings)
    ffmpeg: FFmpegSettings = Field(default_factory=FFmpegSettings)


# Global settings instance (lazy-loaded singleton)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance.

    Returns:
        Settings: The global settings instance.

    Example:
        >>> settings = get_settings()
        >>> print(settings.rabbitmq.host)
        localhost
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment variables.

    This is useful for testing or when environment variables change.

    Returns:
        Settings: The newly loaded settings instance.
    """
    global _settings
    _settings = Settings()
    return _settings

