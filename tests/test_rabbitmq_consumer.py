"""Unit tests for RabbitMQ consumer (with mocks)."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.poc.models import TextMessage
from src.poc.rabbitmq_consumer import RabbitMQConsumer

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestRabbitMQConsumer:
    """Test RabbitMQConsumer class."""

    def test_consumer_initialization(self, mock_settings: None) -> None:
        """Test consumer creation."""
        consumer = RabbitMQConsumer()

        assert consumer.max_reconnect_attempts == 10
        assert consumer.reconnect_delay == 5.0
        assert consumer.message_handler is None

    def test_consumer_initialization_with_handler(
        self, mock_settings: None
    ) -> None:
        """Test consumer creation with message handler."""
        handler = MagicMock()

        consumer = RabbitMQConsumer(message_handler=handler)

        assert consumer.message_handler is handler

    @pytest.mark.asyncio
    @patch("src.poc.rabbitmq_consumer.aio_pika.connect_robust")
    async def test_consumer_connect_success(
        self, mock_connect: MagicMock, mock_settings: None
    ) -> None:
        """Test successful connection."""
        # Mock connection objects
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()

        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_queue.return_value = mock_queue

        consumer = RabbitMQConsumer()
        await consumer.connect()

        assert consumer._connection is not None
        assert consumer._channel is not None
        assert consumer._queue is not None
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.poc.rabbitmq_consumer.aio_pika.connect_robust")
    async def test_consumer_connect_retry(
        self, mock_connect: MagicMock, mock_settings: None
    ) -> None:
        """Test connection retry logic."""
        # First call fails, second succeeds
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()

        mock_connect.side_effect = [
            Exception("Connection failed"),
            mock_connection,
        ]

        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_queue.return_value = mock_queue

        consumer = RabbitMQConsumer(max_reconnect_attempts=3, reconnect_delay=0.1)

        await consumer.connect()

        assert consumer._connection is not None
        assert mock_connect.call_count == 2

    @pytest.mark.asyncio
    @patch("src.poc.rabbitmq_consumer.aio_pika.connect_robust")
    async def test_consumer_connect_max_retries(
        self, mock_connect: MagicMock, mock_settings: None
    ) -> None:
        """Test max retry attempts."""
        mock_connect.side_effect = Exception("Connection failed")

        consumer = RabbitMQConsumer(max_reconnect_attempts=2, reconnect_delay=0.1)

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await consumer.connect()

        assert mock_connect.call_count == 2

    @pytest.mark.asyncio
    @patch("src.poc.rabbitmq_consumer.aio_pika.connect_robust")
    async def test_consumer_context_manager(
        self, mock_connect: MagicMock, mock_settings: None
    ) -> None:
        """Test async context manager."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()

        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_queue.return_value = mock_queue
        mock_connection.is_closed = False
        mock_channel.is_closed = False

        async with RabbitMQConsumer() as consumer:
            assert consumer._connection is not None

        # Verify cleanup
        mock_connection.close.assert_called_once()
        mock_channel.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.poc.rabbitmq_consumer.aio_pika.connect_robust")
    async def test_consumer_message_parsing(
        self, mock_connect: MagicMock, mock_settings: None
    ) -> None:
        """Test message parsing from queue."""
        import json

        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_message = AsyncMock()

        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_queue.return_value = mock_queue

        # Create test message
        test_data = {
            "type": "text",
            "data": "Hello, world!",
            "session_id": "test-session",
        }
        mock_message.body = json.dumps(test_data).encode("utf-8")
        mock_message.ack = AsyncMock()
        mock_message.nack = AsyncMock()

        consumer = RabbitMQConsumer()
        await consumer.connect()

        # Test message handling
        await consumer._handle_message(mock_message)

        # Verify message was acked (successful processing)
        mock_message.ack.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.poc.rabbitmq_consumer.aio_pika.connect_robust")
    async def test_consumer_message_nack_on_error(
        self, mock_connect: MagicMock, mock_settings: None
    ) -> None:
        """Test message nack on parsing error."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_message = AsyncMock()

        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_queue.return_value = mock_queue

        # Invalid message (not valid JSON)
        mock_message.body = b"invalid json"
        mock_message.ack = AsyncMock()
        mock_message.nack = AsyncMock()

        consumer = RabbitMQConsumer()
        await consumer.connect()

        await consumer._handle_message(mock_message)

        # Verify message was nacked (invalid message)
        mock_message.nack.assert_called_once_with(requeue=False)

    @pytest.mark.asyncio
    @patch("src.poc.rabbitmq_consumer.aio_pika.connect_robust")
    async def test_consumer_graceful_shutdown(
        self, mock_connect: MagicMock, mock_settings: None
    ) -> None:
        """Test graceful shutdown."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()

        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_queue.return_value = mock_queue
        mock_connection.is_closed = False
        mock_channel.is_closed = False
        mock_queue.cancel = AsyncMock()

        consumer = RabbitMQConsumer()
        await consumer.connect()
        consumer._consumer_tag = "test-tag"

        await consumer.close()

        assert consumer._running is False
        mock_queue.cancel.assert_called_once_with("test-tag")

