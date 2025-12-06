"""Unit tests for configuration settings."""

import os
from typing import TYPE_CHECKING

import pytest

# Import after fixtures are set up
from src.config.settings import Settings

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestSettings:
    """Test configuration settings."""

    def test_settings_loading(self, monkeypatch: "MonkeyPatch") -> None:
        """Test settings load from defaults."""
        # Ensure RTMP_URL is set before importing
        monkeypatch.setenv("RTMP_URL", "rtmp://test.example.com/live/test")
        # Clear settings cache
        import src.config.settings as settings_module
        settings_module._settings = None
        from src.config import get_settings, reload_settings
        reload_settings()
        settings = get_settings()

        assert settings is not None
        assert settings.app_name == "jj-ai-avatar-livekit-agent-poc"
        assert settings.log_level == "INFO"
        assert settings.debug is False

    def test_rabbitmq_settings(self) -> None:
        """Test RabbitMQ settings."""
        from src.config import get_settings
        settings = get_settings()

        assert settings.rabbitmq.host == "localhost"
        assert settings.rabbitmq.port == 5672
        assert settings.rabbitmq.user == "guest"
        assert settings.rabbitmq.queue == "talking_face_input"

    def test_rabbitmq_connection_url(self) -> None:
        """Test RabbitMQ connection URL generation."""
        from src.config import get_settings
        settings = get_settings()

        url = settings.rabbitmq.connection_url
        assert url.startswith("amqp://")
        assert "guest:guest@" in url
        assert "localhost:5672" in url

    def test_rtmp_settings_required(self, monkeypatch: "MonkeyPatch") -> None:
        """Test RTMP URL is required."""
        from src.config import get_settings, reload_settings
        from pydantic import ValidationError
        # Remove RTMP_URL to test required field
        monkeypatch.delenv("RTMP_URL", raising=False)
        # Clear settings cache
        import src.config.settings as settings_module
        settings_module._settings = None

        with pytest.raises(ValidationError):  # Pydantic validation error
            reload_settings()

    def test_rtmp_settings(self) -> None:
        """Test RTMP settings."""
        from src.config import get_settings
        settings = get_settings()

        assert settings.rtmp.url == "rtmp://test.example.com/live/test"
        assert settings.rtmp.resolution == "1280x720"
        assert settings.rtmp.fps == 30
        assert settings.rtmp.width == 1280
        assert settings.rtmp.height == 720

    def test_rtmp_resolution_validation(self, monkeypatch: "MonkeyPatch") -> None:
        """Test RTMP resolution format validation."""
        from src.config import get_settings, reload_settings
        from pydantic import ValidationError
        monkeypatch.setenv("RTMP_URL", "rtmp://test")
        monkeypatch.setenv("RTMP_RESOLUTION", "invalid")
        # Clear settings cache
        import src.config.settings as settings_module
        settings_module._settings = None

        with pytest.raises(ValidationError):  # Validation error
            reload_settings()

    def test_rtmp_resolution_properties(self) -> None:
        """Test RTMP resolution width/height properties."""
        from src.config import get_settings
        settings = get_settings()

        assert settings.rtmp.width == 1280
        assert settings.rtmp.height == 720

        # Test with different resolution
        settings.rtmp.resolution = "1920x1080"
        assert settings.rtmp.width == 1920
        assert settings.rtmp.height == 1080

    def test_static_video_settings(self) -> None:
        """Test static video settings."""
        from src.config import get_settings
        settings = get_settings()

        assert settings.static_video.loop is True
        assert settings.static_video.fps is None  # Uses RTMP FPS by default

    def test_tts_settings(self) -> None:
        """Test TTS settings."""
        from src.config import get_settings
        settings = get_settings()

        assert settings.tts.provider == "local"
        assert settings.tts.local.provider == "edge-tts"
        assert settings.tts.local.language == "en"
        assert settings.tts.local.sample_rate == 44100
        assert settings.tts.local.channels == 1

    def test_settings_environment_variables(
        self, monkeypatch: "MonkeyPatch"
    ) -> None:
        """Test settings load from environment variables."""
        from src.config import get_settings, reload_settings
        monkeypatch.setenv("RTMP_URL", "rtmp://custom.example.com/live/key")
        monkeypatch.setenv("RABBITMQ_HOST", "custom-host")
        monkeypatch.setenv("RABBITMQ_PORT", "5673")
        monkeypatch.setenv("TTS_PROVIDER", "local")
        monkeypatch.setenv("TTS_LOCAL_LANGUAGE", "es")
        # Clear settings cache
        import src.config.settings as settings_module
        settings_module._settings = None

        reload_settings()
        settings = get_settings()

        assert settings.rtmp.url == "rtmp://custom.example.com/live/key"
        assert settings.rabbitmq.host == "custom-host"
        assert settings.rabbitmq.port == 5673
        assert settings.tts.local.language == "es"

    def test_settings_singleton(self) -> None:
        """Test get_settings() singleton pattern."""
        from src.config import get_settings
        settings1 = get_settings()
        settings2 = get_settings()

        # Should return the same instance
        assert settings1 is settings2

    def test_settings_reload(self, monkeypatch: "MonkeyPatch") -> None:
        """Test reload_settings() function."""
        from src.config import get_settings, reload_settings
        monkeypatch.setenv("RTMP_URL", "rtmp://test.example.com/live/test")
        reload_settings()
        settings1 = get_settings()

        # Reload should create new instance
        settings2 = reload_settings()

        # Should be different instances
        assert settings1 is not settings2

        # But new calls should return the new instance
        settings3 = get_settings()
        assert settings2 is settings3

