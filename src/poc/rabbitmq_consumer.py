"""Async RabbitMQ consumer for PoC application.

This module provides an async RabbitMQ consumer with automatic reconnection,
retry logic, and graceful shutdown support.
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Awaitable, Callable, Optional, Union

import aio_pika
from aio_pika import IncomingMessage, Message as RabbitMQMessage
from aio_pika.abc import AbstractConnection, AbstractChannel, AbstractQueue

from src.config import get_settings
from src.poc.models import Message

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    """Async RabbitMQ consumer with automatic reconnection.

    This consumer connects to RabbitMQ, subscribes to a queue, and
    processes messages with automatic reconnection on failures.

    Example:
        ```python
        async with RabbitMQConsumer() as consumer:
            async for message in consumer.consume():
                # Process message
                print(f"Received: {message.data}")
        ```
    """

    def __init__(
        self,
        message_handler: Optional[
            Union[
                Callable[[Message], None],
                Callable[[Message], Awaitable[None]],
            ]
        ] = None,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 5.0,
    ) -> None:
        """Initialize RabbitMQ consumer.

        Args:
            message_handler: Optional callback function to handle messages.
                            If provided, messages will be passed to this handler.
                            If None, messages are yielded from consume() iterator.
            max_reconnect_attempts: Maximum number of reconnection attempts.
            reconnect_delay: Delay between reconnection attempts in seconds.
        """
        self.settings = get_settings().rabbitmq
        self.message_handler = message_handler
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._connection: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._queue: Optional[AbstractQueue] = None
        self._consumer_tag: Optional[str] = None
        self._running = False
        self._reconnect_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "RabbitMQConsumer":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to RabbitMQ server with retry logic.

        Raises:
            ConnectionError: If connection fails after max attempts.
        """
        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                logger.info(
                    f"Connecting to RabbitMQ at {self.settings.host}:{self.settings.port} "
                    f"(attempt {attempt}/{self.max_reconnect_attempts})"
                )

                self._connection = await aio_pika.connect_robust(
                    self.settings.connection_url,
                    client_properties={"connection_name": "jj-ai-avatar-poc"},
                )

                self._channel = await self._connection.channel()
                await self._channel.set_qos(prefetch_count=1)

                # Declare queue (creates if doesn't exist)
                self._queue = await self._channel.declare_queue(
                    self.settings.queue,
                    durable=True,  # Queue survives broker restart
                )

                logger.info(f"Connected to RabbitMQ and subscribed to queue: {self.settings.queue}")
                return

            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ (attempt {attempt}): {e}")
                if attempt < self.max_reconnect_attempts:
                    logger.info(f"Retrying in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to RabbitMQ after {self.max_reconnect_attempts} attempts"
                    ) from e

    async def close(self) -> None:
        """Close RabbitMQ connection gracefully."""
        self._running = False

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._consumer_tag:
            try:
                await self._queue.cancel(self._consumer_tag)
            except Exception as e:
                logger.warning(f"Error cancelling consumer: {e}")

        if self._channel and not self._channel.is_closed:
            await self._channel.close()

        if self._connection and not self._connection.is_closed:
            await self._connection.close()

        logger.info("RabbitMQ connection closed")

    async def _handle_message(self, message: IncomingMessage) -> None:
        """Handle incoming message from RabbitMQ.

        Args:
            message: Incoming message from RabbitMQ.
        """
        try:
            # Parse message body as JSON
            try:
                body = message.body.decode("utf-8")
                data = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse message body: {e}")
                logger.debug(f"Message body: {message.body[:100]}...")
                await message.nack(requeue=False)  # Don't requeue invalid messages
                return

            # Parse into typed message model
            try:
                parsed_message = Message.parse_json(data)
            except ValueError as e:
                logger.error(f"Failed to parse message: {e}")
                logger.debug(f"Message data: {data}")
                await message.nack(requeue=False)  # Don't requeue invalid messages
                return

            logger.info(
                f"Received {parsed_message.type} message "
                f"(session_id: {parsed_message.session_id})"
            )

            # Handle message
            if self.message_handler:
                try:
                    # Call handler (can be sync or async)
                    if asyncio.iscoroutinefunction(self.message_handler):
                        await self.message_handler(parsed_message)
                    else:
                        self.message_handler(parsed_message)
                    # Ack message on successful handling
                    await message.ack()
                except Exception as e:
                    logger.error(f"Error in message handler: {e}", exc_info=True)
                    # Nack and requeue on handler failure
                    await message.nack(requeue=True)
            else:
                # If no handler, message will be yielded from consume() iterator
                # This case shouldn't happen when using _handle_message
                logger.warning("No message handler set, but _handle_message called")
                await message.ack()

        except Exception as e:
            logger.error(f"Unexpected error processing message: {e}", exc_info=True)
            # Nack and requeue on unexpected errors
            await message.nack(requeue=True)

    async def consume(self) -> AsyncIterator[Message]:
        """Consume messages from RabbitMQ queue.

        This is an async iterator that yields parsed messages.
        Use this when no message_handler is provided.

        Yields:
            Message: Parsed message (TextMessage or ControlMessage).

        Example:
            ```python
            async for message in consumer.consume():
                if isinstance(message, TextMessage):
                    print(f"Text: {message.data}")
            ```
        """
        if not self._connection or self._connection.is_closed:
            await self.connect()

        if not self._queue:
            raise RuntimeError("Queue not initialized. Call connect() first.")

        self._running = True
        message_queue: asyncio.Queue[Message] = asyncio.Queue()

        async def message_callback(message: IncomingMessage) -> None:
            """Callback for incoming messages."""
            try:
                # Parse message
                body = message.body.decode("utf-8")
                data = json.loads(body)
                parsed_message = Message.parse_json(data)

                # Put in queue for yielding
                await message_queue.put(parsed_message)

                # Ack message
                await message.ack()
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await message.nack(requeue=True)

        # Start consuming
        self._consumer_tag = await self._queue.consume(message_callback)

        logger.info(f"Started consuming messages from queue: {self.settings.queue}")

        try:
            while self._running:
                try:
                    # Wait for message with timeout to allow checking _running flag
                    message = await asyncio.wait_for(message_queue.get(), timeout=1.0)
                    yield message
                except asyncio.TimeoutError:
                    # Continue loop to check _running flag
                    continue
        finally:
            if self._consumer_tag:
                try:
                    await self._queue.cancel(self._consumer_tag)
                except Exception as e:
                    logger.warning(f"Error cancelling consumer: {e}")

    async def start_consuming(
        self,
        message_handler: Union[
            Callable[[Message], None],
            Callable[[Message], Awaitable[None]],
        ],
    ) -> None:
        """Start consuming messages with a handler function.

        This method runs indefinitely until close() is called.
        Use this when you have a message handler function.

        Args:
            message_handler: Function to handle incoming messages.

        Example:
            ```python
            async def handle_message(message: Message) -> None:
                print(f"Received: {message}")

            await consumer.start_consuming(handle_message)
            ```
        """
        if not self._connection or self._connection.is_closed:
            await self.connect()

        if not self._queue:
            raise RuntimeError("Queue not initialized. Call connect() first.")

        self._running = True
        self.message_handler = message_handler

        # Start consuming
        self._consumer_tag = await self._queue.consume(self._handle_message)

        logger.info(f"Started consuming messages from queue: {self.settings.queue}")

        # Keep running until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
        finally:
            if self._consumer_tag:
                try:
                    await self._queue.cancel(self._consumer_tag)
                except Exception as e:
                    logger.warning(f"Error cancelling consumer: {e}")

    async def _reconnect_loop(self) -> None:
        """Background task for automatic reconnection."""
        while self._running:
            try:
                if self._connection and not self._connection.is_closed:
                    # Connection is healthy, wait a bit
                    await asyncio.sleep(5)
                    continue

                # Connection lost, try to reconnect
                logger.warning("Connection lost, attempting to reconnect...")
                await self.connect()

                # Restart consuming if we were consuming before
                if self._consumer_tag and self._queue:
                    self._consumer_tag = await self._queue.consume(self._handle_message)
                    logger.info("Reconnected and resumed consuming")

            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
                await asyncio.sleep(self.reconnect_delay)

    def start_auto_reconnect(self) -> None:
        """Start automatic reconnection in background.

        This creates a background task that monitors the connection
        and automatically reconnects if it's lost.
        """
        if self._reconnect_task and not self._reconnect_task.done():
            logger.warning("Auto-reconnect already running")
            return

        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        logger.info("Started automatic reconnection")

