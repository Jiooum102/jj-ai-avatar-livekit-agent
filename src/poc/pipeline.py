"""Main pipeline for talking face livestream.

This module orchestrates the complete pipeline:
- RabbitMQ message consumption
- Text-to-Speech conversion
- Talking face generation
- Seamless source switching
- Continuous RTMP streaming
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.config import get_settings
from src.poc.models import TextMessage
from src.poc.rabbitmq_consumer import RabbitMQConsumer
from src.poc.stream_manager import StreamManager
from src.poc.stream_state import StreamState
from src.poc.talking_face.factory import create_talking_face_provider
from src.poc.tts.factory import create_tts_provider

logger = logging.getLogger(__name__)


class TalkingFacePipeline:
    """Main pipeline for talking face livestream.

    This pipeline orchestrates the complete workflow:
    1. Start streaming static video immediately
    2. Consume messages from RabbitMQ
    3. Convert text to audio (TTS)
    4. Generate talking face from audio
    5. Seamlessly switch sources
    6. Return to static video after completion
    """

    def __init__(
        self,
        stream_manager: Optional[StreamManager] = None,
        rabbitmq_consumer: Optional[RabbitMQConsumer] = None,
        tts_provider=None,
        talking_face_provider=None,
    ) -> None:
        """Initialize pipeline.

        Args:
            stream_manager: Stream manager instance. If None, creates new one.
            rabbitmq_consumer: RabbitMQ consumer instance. If None, creates new one.
            tts_provider: TTS provider instance. If None, creates from settings.
            talking_face_provider: Talking face provider instance. If None, creates from settings.
        """
        self.settings = get_settings()

        # Initialize components
        self.stream_manager = stream_manager or StreamManager()
        self.rabbitmq_consumer = rabbitmq_consumer or RabbitMQConsumer(
            message_handler=self._handle_message
        )

        # Initialize TTS and talking face providers
        if tts_provider is None:
            self.tts_provider = create_tts_provider()
        else:
            self.tts_provider = tts_provider

        if talking_face_provider is None:
            self.talking_face_provider = create_talking_face_provider()
        else:
            self.talking_face_provider = talking_face_provider

        # Pipeline control
        self._running = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the pipeline.

        This initializes all components and starts streaming.
        """
        if self._running:
            logger.warning("Pipeline already started")
            return

        logger.info("Starting pipeline...")

        # Initialize TTS provider
        await self.tts_provider.initialize()

        # Initialize talking face provider
        await self.talking_face_provider.initialize()

        # Start stream manager (starts RTMP streaming with static video)
        await self.stream_manager.start()

        # Start message processing task
        self._running = True
        self._processing_task = asyncio.create_task(self._message_processing_loop())

        # Start RabbitMQ consumer
        # Connect first if not already connected
        if not self.rabbitmq_consumer._connection or self.rabbitmq_consumer._connection.is_closed:
            await self.rabbitmq_consumer.connect()
        
        # Start consuming with the message handler in background task
        if self.rabbitmq_consumer.message_handler:
            self._consumer_task = asyncio.create_task(
                self.rabbitmq_consumer.start_consuming(self.rabbitmq_consumer.message_handler)
            )
        else:
            # If no handler, start consuming as iterator (shouldn't happen in pipeline)
            logger.warning("No message handler set for RabbitMQ consumer")

        logger.info("Pipeline started")

    async def stop(self) -> None:
        """Stop the pipeline and cleanup resources."""
        if not self._running:
            return

        logger.info("Stopping pipeline...")

        self._running = False

        # Stop RabbitMQ consumer
        # Cancel consumer task first
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        # Close RabbitMQ connection
        await self.rabbitmq_consumer.close()

        # Cancel message processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Stop stream manager
        self.stream_manager.stop()

        # Cleanup providers
        await self.talking_face_provider.cleanup()
        await self.tts_provider.cleanup()

        logger.info("Pipeline stopped")

    async def _handle_message(self, message: TextMessage) -> None:
        """Handle incoming RabbitMQ message.

        This is called by the RabbitMQ consumer when a message is received.
        It queues the message for processing.

        Args:
            message: Text message from RabbitMQ.
        """
        logger.info(f"Received message: {message.data[:50]}... (session: {message.session_id})")
        await self._message_queue.put(message)

    async def _message_processing_loop(self) -> None:
        """Main message processing loop.

        This loop processes messages from the queue sequentially,
        ensuring only one message is processed at a time.
        """
        try:
            while self._running:
                try:
                    # Get message from queue (with timeout to allow checking _running)
                    message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    # Timeout is expected, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error in message processing loop: {e}", exc_info=True)
                    # Set error state
                    self.stream_manager.state_manager.set_state(StreamState.ERROR, force=True)
                    # Try to recover
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.debug("Message processing loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in message processing loop: {e}", exc_info=True)

    async def _process_message(self, message: TextMessage) -> None:
        """Process a single message through the complete pipeline.

        Workflow: Text → TTS → Audio → Talking Face Video

        Args:
            message: Text message to process.
        """
        try:
            # Check if we can process (must be IDLE or ERROR)
            current_state = self.stream_manager.state_manager.state
            if current_state not in (StreamState.IDLE, StreamState.ERROR):
                logger.warning(
                    f"Cannot process message in state {current_state.value}, "
                    "queueing for later processing"
                )
                # Re-queue message
                await self._message_queue.put(message)
                return

            # Set state to PROCESSING
            self.stream_manager.state_manager.set_state(StreamState.PROCESSING)

            logger.info(f"Processing message: {message.data[:50]}...")

            # Step 1: Convert text to audio using TTS
            logger.debug("Converting text to speech...")
            audio_bytes = await self.tts_provider.synthesize(
                text=message.data,
                language=message.language or "en",
                voice_id=message.voice_id,
            )

            # Step 2: Generate talking face from audio
            logger.debug("Generating talking face from audio...")

            # Get avatar path from settings
            avatar_path = self.settings.talking_face.musetalk.avatar_image

            # Generate talking face frames
            frame_generator = self.talking_face_provider.generate_from_audio(
                audio=audio_bytes,
                avatar=avatar_path,
                fps=self.settings.rtmp.fps,
                resolution=(self.settings.rtmp.width, self.settings.rtmp.height),
            )

            # Extract audio from bytes for synchronized playback
            # Convert audio bytes to numpy array for frame-by-frame playback
            audio_buffer = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_buffer)
            audio_data = audio_data.astype(np.float32)

            # Convert to int16 for RTMP
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767.0).astype(np.int16)

            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Create audio generator synchronized with frames
            # Calculate samples per frame
            samples_per_frame = int(sample_rate / self.settings.rtmp.fps)
            total_frames = int(len(audio_data) / samples_per_frame)

            async def audio_generator():
                """Generate audio chunks synchronized with frames."""
                for i in range(total_frames):
                    start_idx = i * samples_per_frame
                    end_idx = min(start_idx + samples_per_frame, len(audio_data))
                    if start_idx < len(audio_data):
                        yield audio_data[start_idx:end_idx]
                    else:
                        # Silence for remaining frames
                        yield np.zeros(samples_per_frame, dtype=np.int16)

            # Step 3: Switch to talking face source
            await self.stream_manager.switch_to_talking_face(frame_generator, audio_generator())

            # Step 4: Wait for talking face to complete
            # We need to wait until all frames are consumed
            # This is handled by the stream manager and buffer

            # For now, we'll wait a reasonable time based on audio length
            audio_duration = len(audio_data) / sample_rate
            # Add some buffer time
            wait_time = audio_duration + 1.0
            logger.debug(f"Waiting {wait_time:.2f}s for talking face to complete...")
            await asyncio.sleep(wait_time)

            # Step 5: Switch back to static video
            await self.stream_manager.switch_to_static_video()

            logger.info(f"Message processing completed: {message.session_id}")

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Set error state
            self.stream_manager.state_manager.set_state(StreamState.ERROR, force=True)

            # Try to recover by switching back to static video
            try:
                await self.stream_manager.switch_to_static_video()
            except Exception as recovery_error:
                logger.error(f"Error recovering to static video: {recovery_error}")

    def get_statistics(self) -> dict:
        """Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics.
        """
        return {
            "running": self._running,
            "message_queue_size": self._message_queue.qsize(),
            "stream_manager": self.stream_manager.get_statistics(),
        }

