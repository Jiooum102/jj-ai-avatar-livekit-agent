"""Main application entry point for PoC talking face livestream.

This module provides the CLI interface and orchestrates the complete
application lifecycle: initialization, startup, runtime, and graceful shutdown.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from src.config import get_settings, reload_settings
from src.poc.pipeline import TalkingFacePipeline

logger = logging.getLogger(__name__)


class Application:
    """Main application class for PoC talking face livestream."""

    def __init__(self) -> None:
        """Initialize application."""
        self.pipeline: Optional[TalkingFacePipeline] = None
        self.shutdown_event = asyncio.Event()
        self._shutdown_requested = False

    async def start(self) -> None:
        """Start the application.

        This initializes all components and starts the pipeline.
        """
        logger.info("Starting PoC talking face livestream application...")

        # Load configuration (env_file is handled in main() function)
        # Note: This will use the global settings instance which may have been
        # initialized with a custom env_file in main()
        settings = get_settings()
        logger.info(f"Configuration loaded: {settings.app_name}")

        # Create pipeline (uses factories internally for TTS and talking face)
        self.pipeline = TalkingFacePipeline()

        # Start pipeline (this initializes all components and starts streaming)
        await self.pipeline.start()

        logger.info("Application started successfully")
        logger.info("Streaming static video to RTMP server")
        logger.info("Waiting for messages from RabbitMQ...")

    async def stop(self) -> None:
        """Stop the application gracefully."""
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        logger.info("Shutting down application...")

        if self.pipeline:
            try:
                await self.pipeline.stop()
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")

        logger.info("Application stopped")

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            # Set shutdown event (thread-safe)
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self.shutdown_event.set)
            except RuntimeError:
                # If no event loop is running, just set the event
                self.shutdown_event.set()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run(self) -> None:
        """Run the application main loop."""
        try:
            await self.start()

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, shutting down...")
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            raise
        finally:
            await self.stop()


def setup_logging(log_level: Optional[str] = None) -> None:
    """Configure logging for the application.

    Args:
        log_level: Optional log level override. If None, uses settings.log_level.
    """
    if log_level is None:
        try:
            settings = get_settings()
            log_level = settings.log_level
        except Exception:
            log_level = "INFO"

    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific loggers
    logging.getLogger("src.poc").setLevel(numeric_level)
    logging.getLogger("src.config").setLevel(numeric_level)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="PoC Talking Face Livestream - RabbitMQ to RTMP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m src.poc.main

  # Run with custom log level
  python -m src.poc.main --log-level DEBUG

  # Test configuration without starting stream
  python -m src.poc.main --dry-run

  # Use custom environment file
  python -m src.poc.main --env-file .env.production
        """,
    )

    parser.add_argument(
        "--env-file",
        "--config",
        type=str,
        default=None,
        help="Path to environment file (default: .env)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override log level (default: from settings)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test configuration without starting stream",
    )

    return parser.parse_args()


async def main() -> None:
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("PoC Talking Face Livestream Application")
    logger.info("=" * 60)

    # Handle environment file override
    env_file = None
    if args.env_file:
        env_path = Path(args.env_file)
        if not env_path.exists():
            logger.error(f"Environment file not found: {env_path}")
            sys.exit(1)
        env_file = str(env_path)
        logger.info(f"Using environment file: {env_file}")
        # Set ENV_FILE environment variable so subsequent get_settings() calls use it
        import os
        os.environ["ENV_FILE"] = env_file

    # Dry run mode: just test configuration
    if args.dry_run:
        logger.info("DRY RUN MODE: Testing configuration...")
        try:
            settings = get_settings(env_file=env_file)
            logger.info("Configuration loaded successfully:")
            logger.info(f"  App Name: {settings.app_name}")
            logger.info(f"  Log Level: {settings.log_level}")
            logger.info(f"  RabbitMQ: {settings.rabbitmq.host}:{settings.rabbitmq.port}")
            logger.info(f"  RTMP URL: {settings.rtmp.url}")
            logger.info(f"  TTS Provider: {settings.tts.provider}")
            logger.info(f"  Talking Face Provider: {settings.talking_face.provider}")
            logger.info("Configuration test passed!")
            return
        except Exception as e:
            logger.error(f"Configuration test failed: {e}", exc_info=True)
            sys.exit(1)

    # Create and run application
    # Note: env_file is already handled above via get_settings(env_file=env_file)
    get_settings(env_file=env_file)
    app = Application()
    app.setup_signal_handlers()

    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application exited")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

