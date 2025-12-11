"""Unit tests for talking face providers."""

import io
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.poc.talking_face.api_provider import APITalkingFaceProvider
from src.poc.talking_face.base import (
    TalkingFaceModelError,
    TalkingFaceProvider,
    TalkingFaceProviderError,
)
from src.poc.talking_face.factory import create_talking_face_provider, health_check
from src.poc.talking_face.local_provider import LocalTalkingFaceProvider
from src.poc.talking_face.model_manager import ModelManager
from src.poc.talking_face.models.musetalk import MuseTalkModel

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestTalkingFaceProviderBase:
    """Test base TalkingFaceProvider interface."""

    def test_provider_interface(self) -> None:
        """Test that provider interface is properly defined."""
        # Verify abstract methods exist
        assert hasattr(TalkingFaceProvider, "generate_from_audio")
        assert hasattr(TalkingFaceProvider, "generate_video_to_video")
        assert hasattr(TalkingFaceProvider, "initialize")
        assert hasattr(TalkingFaceProvider, "cleanup")
        assert hasattr(TalkingFaceProvider, "is_initialized")


class TestAPITalkingFaceProvider:
    """Test API-based talking face provider."""

    def test_provider_initialization(self) -> None:
        """Test API provider creation."""
        provider = APITalkingFaceProvider(
            url="https://api.example.com",
            api_key="test-key",
            avatar_id="avatar-123",
        )

        assert provider.url == "https://api.example.com"
        assert provider.api_key == "test-key"
        assert provider.avatar_id == "avatar-123"
        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_provider_initialize_success(self) -> None:
        """Test successful provider initialization."""
        provider = APITalkingFaceProvider(
            url="https://api.example.com",
            api_key="test-key",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            await provider.initialize()

            assert provider.is_initialized

    @pytest.mark.asyncio
    async def test_provider_initialize_failure(self) -> None:
        """Test provider initialization failure."""
        provider = APITalkingFaceProvider(
            url="https://api.example.com",
            api_key="test-key",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception(
                "Connection failed"
            )

            with pytest.raises(TalkingFaceProviderError):
                await provider.initialize()

            assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_provider_cleanup(self) -> None:
        """Test provider cleanup."""
        provider = APITalkingFaceProvider(
            url="https://api.example.com",
            api_key="test-key",
        )
        provider._initialized = True

        await provider.cleanup()

        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_generate_from_audio_not_initialized(self) -> None:
        """Test generation fails when not initialized."""
        provider = APITalkingFaceProvider(
            url="https://api.example.com",
            api_key="test-key",
        )

        audio = b"fake audio data"
        avatar = Path("/tmp/test_avatar.png")

        with pytest.raises(TalkingFaceProviderError, match="not initialized"):
            async for _ in provider.generate_from_audio(audio, avatar):
                pass


class TestLocalTalkingFaceProvider:
    """Test local talking face provider."""

    def test_provider_initialization(self) -> None:
        """Test local provider creation."""
        provider = LocalTalkingFaceProvider(
            model_type="musetalk",
            checkpoint_path=Path("./models/musetalk"),
            avatar_image=Path("./assets/avatar.png"),
            device="cuda",
        )

        assert provider.model_type == "musetalk"
        assert provider.checkpoint_path == Path("./models/musetalk")
        assert provider.avatar_image == Path("./assets/avatar.png")
        assert provider.device == "cuda"
        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_provider_initialize_success(self) -> None:
        """Test successful provider initialization."""
        provider = LocalTalkingFaceProvider(
            model_type="musetalk",
            checkpoint_path=Path("./models/musetalk"),
            avatar_image=Path("./assets/avatar.png"),
        )

        with patch("src.poc.talking_face.local_provider.ModelManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.load_model = AsyncMock()
            mock_manager.get_model.return_value = MagicMock()
            mock_manager_class.return_value = mock_manager

            # Create a temporary avatar image for testing
            test_avatar = Path("/tmp/test_avatar.png")
            test_avatar.parent.mkdir(parents=True, exist_ok=True)
            # Create a dummy image file
            import cv2

            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(test_avatar), dummy_image)

            provider.avatar_image = test_avatar

            await provider.initialize()

            assert provider.is_initialized
            assert provider._model_manager is not None

            # Cleanup
            test_avatar.unlink()

    @pytest.mark.asyncio
    async def test_provider_cleanup(self) -> None:
        """Test provider cleanup."""
        provider = LocalTalkingFaceProvider(
            model_type="musetalk",
            checkpoint_path=Path("./models/musetalk"),
        )

        mock_manager = AsyncMock()
        mock_manager.unload_model = AsyncMock()
        provider._model_manager = mock_manager
        provider._initialized = True

        await provider.cleanup()

        assert not provider.is_initialized
        mock_manager.unload_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_from_audio_not_initialized(self) -> None:
        """Test generation fails when not initialized."""
        provider = LocalTalkingFaceProvider(
            model_type="musetalk",
            checkpoint_path=Path("./models/musetalk"),
        )

        audio = b"fake audio data"
        avatar = Path("/tmp/test_avatar.png")

        with pytest.raises(TalkingFaceProviderError, match="not initialized"):
            async for _ in provider.generate_from_audio(audio, avatar):
                pass


class TestModelManager:
    """Test model manager."""

    @pytest.mark.asyncio
    async def test_model_manager_initialization(self) -> None:
        """Test model manager creation."""
        manager = ModelManager(
            model_type="musetalk",
            checkpoint_path=Path("./models/musetalk"),
            device="cuda",
        )

        assert manager.model_type == "musetalk"
        assert manager.checkpoint_path == Path("./models/musetalk")
        assert not manager.is_loaded

    @pytest.mark.asyncio
    async def test_model_manager_load_musetalk(self) -> None:
        """Test loading MuseTalk model."""
        manager = ModelManager(
            model_type="musetalk",
            checkpoint_path=Path("/tmp/test_checkpoint"),
            device="cuda",
        )

        # Create dummy checkpoint directory
        manager.checkpoint_path.mkdir(parents=True, exist_ok=True)

        with patch("src.poc.talking_face.model_manager.MuseTalkModel") as mock_model_class:
            mock_model = AsyncMock()
            mock_model.load = AsyncMock()
            mock_model_class.return_value = mock_model

            await manager.load_model()

            assert manager.is_loaded
            assert manager.get_model() == mock_model

        # Cleanup
        manager.checkpoint_path.rmdir()

    @pytest.mark.asyncio
    async def test_model_manager_load_unknown_model(self) -> None:
        """Test loading unknown model type."""
        manager = ModelManager(
            model_type="unknown_model",
            checkpoint_path=Path("./models/unknown"),
        )

        with pytest.raises(TalkingFaceModelError, match="Unknown model type"):
            await manager.load_model()

    @pytest.mark.asyncio
    async def test_model_manager_unload(self) -> None:
        """Test unloading model."""
        manager = ModelManager(
            model_type="musetalk",
            checkpoint_path=Path("./models/musetalk"),
        )

        mock_model = AsyncMock()
        mock_model.unload = AsyncMock()
        manager._model = mock_model

        await manager.unload_model()

        assert not manager.is_loaded
        assert manager._model is None
        mock_model.unload.assert_called_once()

    def test_model_manager_get_model_not_loaded(self) -> None:
        """Test getting model when not loaded."""
        manager = ModelManager(
            model_type="musetalk",
            checkpoint_path=Path("./models/musetalk"),
        )

        with pytest.raises(TalkingFaceModelError, match="not loaded"):
            manager.get_model()


class TestMuseTalkModel:
    """Test MuseTalk model integration."""

    def test_musetalk_model_initialization(self) -> None:
        """Test MuseTalk model creation."""
        model = MuseTalkModel(
            checkpoint_path=Path("./models/musetalk"),
            device="cuda",
            batch_size=1,
        )

        assert model.checkpoint_path == Path("./models/musetalk")
        assert model.device_str == "cuda"
        assert model.batch_size == 1
        assert not model.is_loaded

    @pytest.mark.asyncio
    async def test_musetalk_model_load_failure_no_checkpoint(self) -> None:
        """Test model loading fails when checkpoint doesn't exist."""
        model = MuseTalkModel(
            checkpoint_path=Path("/nonexistent/path"),
            device="cpu",
        )

        with pytest.raises(TalkingFaceModelError, match="checkpoint not found"):
            await model.load()

    @pytest.mark.asyncio
    async def test_musetalk_model_unload(self) -> None:
        """Test model unloading."""
        model = MuseTalkModel(
            checkpoint_path=Path("./models/musetalk"),
            device="cpu",
        )
        model._loaded = True
        model._vae = object()
        model._unet = object()

        await model.unload()

        assert not model.is_loaded
        assert model._vae is None
        assert model._unet is None

    def test_musetalk_model_not_available(self) -> None:
        """Test graceful handling when MuseTalk is not available."""
        # This test verifies that the model can be created even if MuseTalk modules aren't available
        # The actual loading will fail, but initialization should work
        model = MuseTalkModel(
            checkpoint_path=Path("./models/musetalk"),
            device="cpu",
        )

        # Model should be created but not loaded
        assert not model.is_loaded

    @pytest.mark.asyncio
    async def test_musetalk_generate_not_loaded(self) -> None:
        """Test generation fails when model not loaded."""
        model = MuseTalkModel(
            checkpoint_path=Path("./models/musetalk"),
            device="cpu",
        )

        audio = b"fake audio data"
        avatar = Path("/tmp/test_avatar.png")

        with pytest.raises(TalkingFaceModelError, match="not loaded"):
            async for _ in model.generate_from_audio(audio, avatar):
                pass


class TestTalkingFaceFactory:
    """Test talking face factory."""

    def test_create_api_provider(self, monkeypatch: "MonkeyPatch") -> None:
        """Test creating API provider from settings."""
        monkeypatch.setenv("TALKING_FACE_PROVIDER", "api")
        monkeypatch.setenv("TALKING_FACE_API_URL", "https://api.example.com")
        monkeypatch.setenv("TALKING_FACE_API_API_KEY", "test-key")
        monkeypatch.setenv("RTMP_URL", "rtmp://test")

        from src.config import reload_settings

        reload_settings()
        settings = reload_settings().talking_face

        provider = create_talking_face_provider(settings)

        assert isinstance(provider, APITalkingFaceProvider)
        assert provider.url == "https://api.example.com"
        assert provider.api_key == "test-key"

    def test_create_local_provider(self, monkeypatch: "MonkeyPatch") -> None:
        """Test creating local provider from settings."""
        monkeypatch.setenv("TALKING_FACE_PROVIDER", "local")
        monkeypatch.setenv("TALKING_FACE_MODEL", "musetalk")
        monkeypatch.setenv("RTMP_URL", "rtmp://test")

        from src.config import reload_settings

        reload_settings()
        settings = reload_settings().talking_face

        provider = create_talking_face_provider(settings)

        assert isinstance(provider, LocalTalkingFaceProvider)
        assert provider.model_type == "musetalk"

    def test_create_provider_missing_api_key(self, monkeypatch: "MonkeyPatch") -> None:
        """Test creating API provider without API key fails."""
        monkeypatch.setenv("TALKING_FACE_PROVIDER", "api")
        monkeypatch.setenv("TALKING_FACE_API_URL", "https://api.example.com")
        monkeypatch.setenv("RTMP_URL", "rtmp://test")

        from src.config import reload_settings

        reload_settings()
        settings = reload_settings().talking_face

        with pytest.raises(TalkingFaceProviderError, match="API_KEY"):
            create_talking_face_provider(settings)

    @pytest.mark.asyncio
    async def test_health_check_initialized(self) -> None:
        """Test health check for initialized provider."""
        provider = APITalkingFaceProvider(
            url="https://api.example.com",
            api_key="test-key",
        )
        provider._initialized = True

        assert await health_check(provider) is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self) -> None:
        """Test health check for uninitialized provider."""
        provider = APITalkingFaceProvider(
            url="https://api.example.com",
            api_key="test-key",
        )

        assert await health_check(provider) is False

