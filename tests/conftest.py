"""Shared pytest fixtures for testing."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def test_data_dir() -> Path:
    """Get path to test data directory.

    Returns:
        Path: Path to data/tests directory.
    """
    return Path(__file__).parent.parent / "data" / "tests"


@pytest.fixture
def test_image_path(test_data_dir: Path) -> Path:
    """Get path to test image file.

    Args:
        test_data_dir: Test data directory fixture.

    Returns:
        Path: Path to test JPG image.
    """
    image_path = test_data_dir / "ROG-Zephyrus-Wallpaper_3840x2400.jpg"
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    return image_path


@pytest.fixture
def test_video_path(test_data_dir: Path) -> Path:
    """Get path to test video file.

    Args:
        test_data_dir: Test data directory fixture.

    Returns:
        Path: Path to test MP4 video.
    """
    video_path = test_data_dir / "425829514-999a6f5b-61dd-48e1-b902-bb3f9cbc7247.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture(autouse=True, scope="function")
def mock_settings(monkeypatch: "MonkeyPatch") -> None:
    """Mock settings for testing.

    This fixture is automatically applied to all tests (autouse=True).

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    # Set minimal required environment variables BEFORE any imports
    monkeypatch.setenv("RTMP_URL", "rtmp://test.example.com/live/test")
    monkeypatch.setenv("RABBITMQ_HOST", "localhost")
    monkeypatch.setenv("RABBITMQ_PORT", "5672")
    
    # Clear the global settings cache to force reload
    import src.config.settings as settings_module
    settings_module._settings = None


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path: Path to temporary directory.
    """
    return tmp_path

