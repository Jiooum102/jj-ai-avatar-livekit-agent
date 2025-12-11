#!/usr/bin/env python3
"""Test script for TTS module.

This script tests the TTS provider independently to verify it can:
- Initialize correctly
- Synthesize text to speech
- Generate valid audio output

Usage:
    python scripts/test_tts.py
    python scripts/test_tts.py --text "Custom test message"
    python scripts/test_tts.py --env-file .env.dev
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.poc.tts.factory import create_tts_provider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_tts(text: str = "Hello, this is a test message for TTS synthesis.") -> None:
    """Test TTS provider.

    Args:
        text: Text to synthesize.
    """
    print("=" * 60)
    print("TTS Module Test")
    print("=" * 60)
    print()

    # Load settings
    settings = get_settings()
    tts_settings = settings.tts

    print(f"TTS Provider: {tts_settings.provider}")
    print(f"Language: {tts_settings.local.language}")
    print(f"Voice: {tts_settings.local.voice or 'auto-selected'}")
    print(f"Sample Rate: {tts_settings.local.sample_rate} Hz")
    print(f"Channels: {tts_settings.local.channels}")
    print()

    try:
        # Step 1: Create TTS provider
        print("Step 1: Creating TTS provider...")
        tts_provider = create_tts_provider()
        print("✓ TTS provider created")
        print()

        # Step 2: Initialize TTS provider
        print("Step 2: Initializing TTS provider...")
        await tts_provider.initialize()
        print("✓ TTS provider initialized")
        print()

        # Step 3: List available voices (optional, for debugging)
        if hasattr(tts_provider, "list_voices"):
            print("Step 3: Listing available voices...")
            try:
                voices = await tts_provider.list_voices(language=tts_settings.local.language)
                print(f"✓ Found {len(voices)} voices for language '{tts_settings.local.language}'")
                if voices:
                    print(f"  First voice: {voices[0].get('name', 'Unknown')} ({voices[0].get('id', 'Unknown')})")
            except Exception as e:
                print(f"⚠ Warning: Could not list voices: {e}")
            print()

        # Step 4: Synthesize text to speech
        print(f"Step 4: Synthesizing text to speech...")
        print(f"  Text: {text}")
        print()

        import time
        start_time = time.time()
        audio_bytes = await tts_provider.synthesize(
            text=text,
            language=tts_settings.local.language,
            voice_id=tts_settings.local.voice,
        )
        synthesis_time = time.time() - start_time

        print(f"✓ Audio synthesized in {synthesis_time:.2f} seconds")
        print(f"  Audio size: {len(audio_bytes)} bytes ({len(audio_bytes) / 1024:.2f} KB)")
        print()

        # Step 5: Verify audio format
        print("Step 5: Verifying audio format...")
        import io
        audio_buffer = io.BytesIO(audio_bytes)
        try:
            audio_data, sample_rate = sf.read(audio_buffer)
            duration = len(audio_data) / sample_rate
            channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]

            print(f"✓ Audio format verified")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {channels}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Shape: {audio_data.shape}")
            print()
        except Exception as e:
            print(f"✗ Failed to verify audio format: {e}")
            raise

        # Step 6: Save audio to file
        print("Step 6: Saving audio to file...")
        output_dir = project_root / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "test_tts.wav"

        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        print(f"✓ Audio saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")
        print()

        # Step 7: Cleanup
        print("Step 7: Cleaning up...")
        await tts_provider.cleanup()
        print("✓ Cleanup completed")
        print()

        print("=" * 60)
        print("✓ TTS TEST PASSED")
        print("=" * 60)
        print()
        print(f"Audio file saved to: {output_file}")
        print(f"You can play it with: aplay {output_file} (Linux) or open {output_file} (macOS)")

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ TTS TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test TTS module independently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="Hello, this is a test message for TTS synthesis.",
        help="Text to synthesize (default: 'Hello, this is a test message for TTS synthesis.')",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to environment file (default: .env or ENV_FILE env var)",
    )

    args = parser.parse_args()

    # Set dummy RTMP_URL if not set (test script doesn't need it, but Settings requires it)
    import os
    if "RTMP_URL" not in os.environ:
        os.environ["RTMP_URL"] = "rtmp://dummy.test/live/demo"

    # Reload settings with optional env_file
    if args.env_file:
        from src.config import reload_settings
        reload_settings(env_file=args.env_file)

    try:
        await test_tts(text=args.text)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

