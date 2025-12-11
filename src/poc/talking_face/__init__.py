"""Talking Face module.

This module provides talking face generation capabilities with support for
both API-based and local AI model providers (MuseTalk, MimicTalk, SyncTalk).

The module ensures all providers return video frames in a consistent format:
- Format: numpy arrays (np.ndarray)
- Color space: RGB
- Frame rate: Configurable (typically 30 FPS)
- Resolution: Configurable (typically 1280x720)
"""

from src.poc.talking_face.api_provider import APITalkingFaceProvider
from src.poc.talking_face.base import (
    TalkingFaceError,
    TalkingFaceModelError,
    TalkingFaceProvider,
    TalkingFaceProviderError,
)
from src.poc.talking_face.factory import create_talking_face_provider, health_check
from src.poc.talking_face.local_provider import LocalTalkingFaceProvider

__all__ = [
    "TalkingFaceProvider",
    "APITalkingFaceProvider",
    "LocalTalkingFaceProvider",
    "create_talking_face_provider",
    "health_check",
    "TalkingFaceError",
    "TalkingFaceProviderError",
    "TalkingFaceModelError",
]

