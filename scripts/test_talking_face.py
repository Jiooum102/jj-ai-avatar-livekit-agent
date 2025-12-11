#!/usr/bin/env python3
"""Test script for talking face module.

This script tests the talking face provider independently to verify it can:
- Initialize correctly
- Load models
- Generate talking face frames from audio

Usage:
    python scripts/test_talking_face.py
    python scripts/test_talking_face.py --audio test_outputs/test_tts.wav
    python scripts/test_talking_face.py --env-file .env.dev
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.poc.talking_face.factory import create_talking_face_provider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_talking_face(audio_file: Path = None) -> None:
    """Test talking face provider.

    Args:
        audio_file: Path to audio file. If None, will try to use test_outputs/test_tts.wav
    """
    print("=" * 60)
    print("Talking Face Module Test")
    print("=" * 60)
    print()

    # Load settings
    settings = get_settings()
    talking_face_settings = settings.talking_face

    print(f"Provider: {talking_face_settings.provider}")
    print(f"Model: {talking_face_settings.model}")
    if talking_face_settings.model == "musetalk":
        musetalk_settings = talking_face_settings.musetalk
        print(f"Checkpoint: {musetalk_settings.checkpoint_path}")
        print(f"Avatar Image: {musetalk_settings.avatar_image}")
        print(f"Device: {musetalk_settings.device}")
        print(f"Batch Size: {musetalk_settings.batch_size}")
        print(f"VAE Type: {musetalk_settings.vae_type}")
        print(f"Version: {musetalk_settings.version}")
    print()

    # Find audio file
    if audio_file is None:
        # Try to use TTS test output
        audio_file = project_root / "test_outputs" / "test_tts.wav"
        if not audio_file.exists():
            print("⚠ No audio file specified and test_outputs/test_tts.wav not found.")
            print("  Please run test_tts.py first or specify --audio <file>")
            sys.exit(1)

    if not audio_file.exists():
        print(f"✗ Audio file not found: {audio_file}")
        sys.exit(1)

    print(f"Audio file: {audio_file}")
    print(f"Audio file size: {audio_file.stat().st_size} bytes")
    print()

    try:
        # Step 1: Create talking face provider
        print("Step 1: Creating talking face provider...")
        talking_face_provider = create_talking_face_provider()
        print("✓ Talking face provider created")
        print()

        # Step 2: Initialize talking face provider
        print("Step 2: Initializing talking face provider...")
        print("  This may take a while as models are loaded...")
        import time
        start_time = time.time()
        await talking_face_provider.initialize()
        init_time = time.time() - start_time
        print(f"✓ Talking face provider initialized in {init_time:.2f} seconds")
        print(f"  Initialized: {talking_face_provider.is_initialized}")
        print()

        # Step 3: Load audio file
        print("Step 3: Loading audio file...")
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        print(f"✓ Audio loaded: {len(audio_bytes)} bytes")
        print()

        # Step 4: Get avatar path
        avatar_path = talking_face_settings.musetalk.avatar_image
        print(f"Step 4: Using avatar image: {avatar_path}")
        if not avatar_path.exists():
            print(f"✗ Avatar image not found: {avatar_path}")
            sys.exit(1)
        print(f"✓ Avatar image found")
        print()

        # Step 5: Generate talking face frames
        print("Step 5: Generating talking face frames...")
        print("  This may take a while depending on audio length...")
        print()

        fps = settings.rtmp.fps
        resolution = (settings.rtmp.width, settings.rtmp.height)

        print(f"  Target FPS: {fps}")
        print(f"  Target Resolution: {resolution[0]}x{resolution[1]}")
        print()

        frame_count = 0
        frames = []
        start_time = time.time()

        try:
            async for frame in talking_face_provider.generate_from_audio(
                audio=audio_bytes,
                avatar=avatar_path,
                fps=fps,
                resolution=resolution,
            ):
                frame_count += 1
                frames.append(frame)

                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  Generated {frame_count} frames ({fps_actual:.1f} fps)...")

                # Limit to first 100 frames for testing
                if frame_count >= 100:
                    print(f"  (Limiting to first 100 frames for testing)")
                    break

        except Exception as e:
            print(f"✗ Error during frame generation: {e}")
            raise

        generation_time = time.time() - start_time
        print()
        print(f"✓ Generated {frame_count} frames in {generation_time:.2f} seconds")
        if generation_time > 0:
            print(f"  Average speed: {frame_count / generation_time:.1f} fps")
        print()

        if frame_count == 0:
            print("✗ No frames generated!")
            sys.exit(1)

        # Step 6: Verify frames
        print("Step 6: Verifying frames...")
        first_frame = frames[0]
        print(f"  Frame shape: {first_frame.shape}")
        print(f"  Frame dtype: {first_frame.dtype}")
        print(f"  Frame min: {first_frame.min()}")
        print(f"  Frame max: {first_frame.max()}")
        print(f"  Expected shape: ({resolution[1]}, {resolution[0]}, 3)")
        print()

        if first_frame.shape != (resolution[1], resolution[0], 3):
            print(f"⚠ Warning: Frame shape mismatch!")
            print(f"  Expected: ({resolution[1]}, {resolution[0]}, 3)")
            print(f"  Got: {first_frame.shape}")
        else:
            print("✓ Frame shape matches expected resolution")
        print()

        # Step 7: Save sample frames
        print("Step 7: Saving sample frames...")
        output_dir = project_root / "test_outputs"
        output_dir.mkdir(exist_ok=True)

        # Save first 10 frames
        num_frames_to_save = min(10, len(frames))
        for i in range(num_frames_to_save):
            frame = frames[i]
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_file = output_dir / f"test_frame_{i:03d}.png"
            cv2.imwrite(str(output_file), frame_bgr)
            print(f"  Saved frame {i+1}/{num_frames_to_save}: {output_file}")

        print()
        print(f"✓ Saved {num_frames_to_save} sample frames")
        print()

        # Step 8: Cleanup
        print("Step 8: Cleaning up...")
        await talking_face_provider.cleanup()
        print("✓ Cleanup completed")
        print()

        print("=" * 60)
        print("✓ TALKING FACE TEST PASSED")
        print("=" * 60)
        print()
        print(f"Generated {frame_count} frames")
        print(f"Sample frames saved to: {output_dir}")
        print(f"You can view them with: open {output_dir}/test_frame_*.png")

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ TALKING FACE TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test talking face module independently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        help="Path to audio file (default: test_outputs/test_tts.wav)",
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
        await test_talking_face(audio_file=args.audio)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

