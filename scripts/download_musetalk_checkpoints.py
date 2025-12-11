#!/usr/bin/env python3
"""Download MuseTalk model checkpoints if they don't exist.

This script downloads the required MuseTalk model checkpoints from HuggingFace,
Google Drive, and other sources. It checks if files already exist before downloading
to avoid re-downloading.

Usage:
    python scripts/download_musetalk_checkpoints.py [--checkpoint-dir DIR] [--version v15|v1]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Error: requests and tqdm are required. Install them with: pip install requests tqdm")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model repositories
MUSETALK_HF_REPO = "TMElyralab/MuseTalk"
VAE_HF_REPO = "stabilityai/sd-vae-ft-mse"
WHISPER_HF_REPO = "openai/whisper-tiny"
DWPOSE_HF_REPO = "yzd-v/DWPose"
SYNCNET_HF_REPO = "ByteDance/LatentSync"

# Face parsing model URLs
RESNET18_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
# BiSeNet model from Google Drive
BISENET_GDOWN_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"

# HuggingFace mirror endpoint (optional, for faster downloads in some regions)
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


def check_command(cmd: str) -> tuple[bool, str]:
    """Check if a command is available.
    
    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        subprocess.run(
            [cmd, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True, ""
    except FileNotFoundError:
        if cmd == "hf":
            return False, "hf (HuggingFace CLI) not found. Install it with: pip install -U 'huggingface_hub[cli]'"
        elif cmd == "gdown":
            return False, "gdown not found. Install it with: pip install gdown"
        else:
            return False, f"{cmd} not found in PATH"
    except subprocess.CalledProcessError:
        return False, f"{cmd} is installed but returned an error"


def download_file(url: str, dest_path: Path, description: str = "Downloading") -> bool:
    """Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        description: Description for progress bar
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, "wb") as f, tqdm(
            desc=description,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"✓ Downloaded: {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def run_hf_cli_download(
    repo_id: str,
    local_dir: Path,
    include_files: list[str],
) -> bool:
    """Download files from HuggingFace using hf CLI.
    
    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to save files
        include_files: List of file patterns to include
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        # Set HF_ENDPOINT environment variable
        env = os.environ.copy()
        env["HF_ENDPOINT"] = HF_ENDPOINT
        
        cmd = [
            "hf",
            "download",
            repo_id,
            "--local-dir",
            str(local_dir),
        ]
        
        for file_pattern in include_files:
            cmd.extend(["--include", file_pattern])
        
        logger.info(f"Downloading from {repo_id} to {local_dir}...")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"✓ Downloaded from {repo_id}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download from {repo_id}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("hf (HuggingFace CLI) not found. Install it with: pip install -U 'huggingface_hub[cli]'")
        return False


def run_gdown(file_id: str, dest_path: Path, description: str = "Downloading") -> bool:
    """Download a file from Google Drive using gdown.
    
    Args:
        file_id: Google Drive file ID
        dest_path: Destination file path
        description: Description for logging
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{description} from Google Drive (ID: {file_id})...")
        cmd = [
            "gdown",
            "--id",
            file_id,
            "-O",
            str(dest_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"✓ Downloaded: {dest_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download from Google Drive: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("gdown not found. Install it with: pip install gdown")
        return False


def download_musetalk_checkpoints(checkpoints_dir: Path, version: str = "v15") -> bool:
    """Download MuseTalk model checkpoints.
    
    Args:
        checkpoints_dir: Base directory for all checkpoints
        version: MuseTalk version ('v15' or 'v1')
        
    Returns:
        True if all required files are present, False otherwise
    """
    logger.info(f"Downloading MuseTalk {version} checkpoints...")
    
    if version == "v15":
        subfolder = "musetalkV15"
        include_files = ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"]
        required_files = {
            checkpoints_dir / "musetalkV15" / "unet.pth": "UNet model weights",
            checkpoints_dir / "musetalkV15" / "musetalk.json": "UNet configuration",
        }
    else:  # v1
        subfolder = "musetalk"
        include_files = ["musetalk/musetalk.json", "musetalk/pytorch_model.bin"]
        required_files = {
            checkpoints_dir / "musetalk" / "pytorch_model.bin": "UNet model weights",
            checkpoints_dir / "musetalk" / "musetalk.json": "UNet configuration",
        }
    
    # Check if files already exist
    all_exist = all(f.exists() for f in required_files.keys())
    if all_exist:
        logger.info(f"✓ MuseTalk {version} checkpoints already exist")
        return True
    
    # Download using hf CLI
    return run_hf_cli_download(
        repo_id=MUSETALK_HF_REPO,
        local_dir=checkpoints_dir,
        include_files=include_files,
    )


def download_vae_model(checkpoints_dir: Path) -> bool:
    """Download VAE model from HuggingFace.
    
    Args:
        checkpoints_dir: Base directory for all checkpoints
        
    Returns:
        True if download succeeded, False otherwise
    """
    logger.info("Downloading VAE model...")
    
    # Use sd-vae to match original MuseTalk implementation
    vae_dir = checkpoints_dir / "sd-vae"
    required_files = [
        vae_dir / "config.json",
        vae_dir / "diffusion_pytorch_model.bin",
    ]
    
    # Check if files already exist
    if all(f.exists() for f in required_files):
        logger.info("✓ VAE model already exists")
        return True
    
    # Download using hf CLI
    return run_hf_cli_download(
        repo_id=VAE_HF_REPO,
        local_dir=vae_dir,
        include_files=["config.json", "diffusion_pytorch_model.bin"],
    )


def download_whisper_model(checkpoints_dir: Path) -> bool:
    """Download Whisper model from HuggingFace.
    
    Args:
        checkpoints_dir: Base directory for all checkpoints
        
    Returns:
        True if download succeeded, False otherwise
    """
    logger.info("Downloading Whisper model...")
    
    whisper_dir = checkpoints_dir / "whisper"
    required_files = [
        whisper_dir / "config.json",
        whisper_dir / "pytorch_model.bin",
        whisper_dir / "preprocessor_config.json",
    ]
    
    # Check if files already exist
    if all(f.exists() for f in required_files):
        logger.info("✓ Whisper model already exists")
        return True
    
    # Download using hf CLI
    return run_hf_cli_download(
        repo_id=WHISPER_HF_REPO,
        local_dir=whisper_dir,
        include_files=["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    )


def download_dwpose_checkpoint(checkpoints_dir: Path) -> bool:
    """Download DWPose checkpoint from HuggingFace.
    
    Args:
        checkpoints_dir: Base directory for all checkpoints
        
    Returns:
        True if download succeeded, False otherwise
    """
    logger.info("Downloading DWPose checkpoint...")
    
    dwpose_dir = checkpoints_dir / "dwpose"
    checkpoint_path = dwpose_dir / "dw-ll_ucoco_384.pth"
    
    if checkpoint_path.exists():
        logger.info("✓ DWPose checkpoint already exists")
        return True
    
    # Download using hf CLI
    return run_hf_cli_download(
        repo_id=DWPOSE_HF_REPO,
        local_dir=dwpose_dir,
        include_files=["dw-ll_ucoco_384.pth"],
    )


def download_syncnet_checkpoint(checkpoints_dir: Path) -> bool:
    """Download SyncNet checkpoint from HuggingFace.
    
    Args:
        checkpoints_dir: Base directory for all checkpoints
        
    Returns:
        True if download succeeded, False otherwise
    """
    logger.info("Downloading SyncNet checkpoint...")
    
    syncnet_dir = checkpoints_dir / "syncnet"
    checkpoint_path = syncnet_dir / "latentsync_syncnet.pt"
    
    if checkpoint_path.exists():
        logger.info("✓ SyncNet checkpoint already exists")
        return True
    
    # Download using hf CLI
    return run_hf_cli_download(
        repo_id=SYNCNET_HF_REPO,
        local_dir=syncnet_dir,
        include_files=["latentsync_syncnet.pt"],
    )


def download_face_parsing_models(checkpoints_dir: Path) -> bool:
    """Download face parsing models (ResNet18 and BiSeNet).
    
    Args:
        checkpoints_dir: Base directory for all checkpoints
        
    Returns:
        True if all downloads succeeded, False otherwise
    """
    logger.info("Downloading face parsing models...")
    
    face_parse_dir = checkpoints_dir / "face-parse-bisent"
    resnet18_path = face_parse_dir / "resnet18-5c106cde.pth"
    bisenet_path = face_parse_dir / "79999_iter.pth"
    
    success = True
    
    # Download ResNet18 weights
    if resnet18_path.exists():
        logger.info("✓ ResNet18 weights already exist")
    else:
        logger.info("Downloading ResNet18 weights from PyTorch...")
        if not download_file(
            url=RESNET18_URL,
            dest_path=resnet18_path,
            description="ResNet18 weights",
        ):
            logger.warning("Failed to download ResNet18 weights")
            success = False
    
    # Download BiSeNet model from Google Drive
    if bisenet_path.exists():
        logger.info("✓ BiSeNet model already exists")
    else:
        logger.info("Downloading BiSeNet face parsing model from Google Drive...")
        if not run_gdown(
            file_id=BISENET_GDOWN_ID,
            dest_path=bisenet_path,
            description="BiSeNet model",
        ):
            logger.warning("Failed to download BiSeNet model")
            success = False
    
    if success:
        logger.info("✓ Face parsing models downloaded")
    
    return success


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download MuseTalk model checkpoints if they don't exist"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./models",
        help="Base directory for all checkpoints (default: ./models)",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v15"],
        default="v15",
        help="MuseTalk version to download (default: v15)",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip downloading Whisper model (optional)",
    )
    parser.add_argument(
        "--skip-dwpose",
        action="store_true",
        help="Skip downloading DWPose checkpoint (optional)",
    )
    parser.add_argument(
        "--skip-syncnet",
        action="store_true",
        help="Skip downloading SyncNet checkpoint (optional)",
    )
    parser.add_argument(
        "--skip-face-parse",
        action="store_true",
        help="Skip downloading face parsing models (not recommended, required for MuseTalk)",
    )
    
    args = parser.parse_args()
    
    checkpoints_dir = Path(args.checkpoint_dir).resolve()
    
    # # Check for required commands
    # hf_available, hf_error = check_command("hf")
    # if not hf_available:
    #     logger.error(f"❌ {hf_error}")
    #     logger.info("\nTo install required dependencies, run:")
    #     logger.info("  pip install -U 'huggingface_hub[cli]'")
    #     if not args.skip_face_parse:
    #         logger.info("  pip install gdown")
    #     logger.info("\nOr install all at once:")
    #     logger.info("  pip install -U 'huggingface_hub[cli]' gdown")
    #     return 1
    
    # if not args.skip_face_parse:
    #     gdown_available, gdown_error = check_command("gdown")
    #     if not gdown_available:
    #         logger.error(f"❌ {gdown_error}")
    #         logger.info("\nTo install gdown, run:")
    #         logger.info("  pip install gdown")
    #         return 1
    
    # Create necessary directories
    directories = [
        checkpoints_dir / "musetalk",
        checkpoints_dir / "musetalkV15",
        checkpoints_dir / "syncnet",
        checkpoints_dir / "dwpose",
        checkpoints_dir / "face-parse-bisent",
        checkpoints_dir / "sd-vae",  # Match original MuseTalk implementation
        checkpoints_dir / "whisper",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("MuseTalk Checkpoint Downloader")
    logger.info("=" * 60)
    logger.info(f"Checkpoint directory: {checkpoints_dir}")
    logger.info(f"Version: {args.version}")
    logger.info(f"HF_ENDPOINT: {HF_ENDPOINT}")
    logger.info("=" * 60)
    
    success = True
    step = 1
    total_steps = 2  # MuseTalk + VAE
    if not args.skip_face_parse:
        total_steps += 1
    if not args.skip_whisper:
        total_steps += 1
    if not args.skip_dwpose:
        total_steps += 1
    if not args.skip_syncnet:
        total_steps += 1
    
    # Download MuseTalk checkpoints
    logger.info(f"\n[{step}/{total_steps}] Downloading MuseTalk checkpoints...")
    if not download_musetalk_checkpoints(checkpoints_dir, args.version):
        logger.warning("MuseTalk checkpoints download failed")
        success = False
    step += 1
    
    # Download VAE model
    logger.info(f"\n[{step}/{total_steps}] Downloading VAE model...")
    if not download_vae_model(checkpoints_dir):
        logger.warning("VAE model download failed")
        success = False
    step += 1
    
    # Download face parsing models (required)
    if not args.skip_face_parse:
        logger.info(f"\n[{step}/{total_steps}] Downloading face parsing models...")
        if not download_face_parsing_models(checkpoints_dir):
            logger.warning("Face parsing models download failed (required for MuseTalk)")
            success = False
        step += 1
    
    # Download Whisper model (optional)
    if not args.skip_whisper:
        logger.info(f"\n[{step}/{total_steps}] Downloading Whisper model...")
        if not download_whisper_model(checkpoints_dir):
            logger.warning("Whisper model download failed (optional, continuing...)")
        step += 1
    
    # Download DWPose (optional)
    if not args.skip_dwpose:
        logger.info(f"\n[{step}/{total_steps}] Downloading DWPose checkpoint...")
        if not download_dwpose_checkpoint(checkpoints_dir):
            logger.warning("DWPose checkpoint download failed (optional, continuing...)")
        step += 1
    
    # Download SyncNet (optional)
    if not args.skip_syncnet:
        logger.info(f"\n[{step}/{total_steps}] Downloading SyncNet checkpoint...")
        if not download_syncnet_checkpoint(checkpoints_dir):
            logger.warning("SyncNet checkpoint download failed (optional, continuing...)")
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ All required checkpoints downloaded successfully!")
        logger.info("\nYou can now run the application with:")
        logger.info(f"  python -m src.poc.main --env-file .env.dev")
        return 0
    else:
        logger.warning("⚠ Some downloads failed. Please check the errors above.")
        logger.info("\nYou may need to download missing files manually.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
