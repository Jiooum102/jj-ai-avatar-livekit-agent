#!/usr/bin/env python3
"""Demo script to send test messages to RabbitMQ for PoC testing.

This script sends test messages to the RabbitMQ queue to verify
the end-to-end pipeline works correctly.

Usage:
    python scripts/demo_poc.py
    python scripts/demo_poc.py --message "Custom message"
    python scripts/demo_poc.py --interactive
    python scripts/demo_poc.py --file messages.json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aio_pika

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings


async def send_message(
    connection: aio_pika.Connection,
    queue_name: str,
    message_data: dict,
) -> None:
    """Send a message to RabbitMQ queue.

    Args:
        connection: RabbitMQ connection.
        queue_name: Queue name to send message to.
        message_data: Message data dictionary.
    """
    channel = await connection.channel()
    queue = await channel.declare_queue(queue_name, durable=True)

    message_body = json.dumps(message_data).encode("utf-8")
    message = aio_pika.Message(
        message_body,
        content_type="application/json",
        timestamp=datetime.now(timezone.utc),
    )

    await channel.default_exchange.publish(message, routing_key=queue_name)
    print(f"✓ Sent message: {message_data.get('data', '')[:50]}...")
    print(f"  Session ID: {message_data.get('session_id')}")

    await channel.close()


async def send_text_message(
    connection: aio_pika.Connection,
    queue_name: str,
    text: str,
    session_id: Optional[str] = None,
    language: str = "en",
    voice_id: Optional[str] = None,
) -> None:
    """Send a text message to RabbitMQ.

    Args:
        connection: RabbitMQ connection.
        queue_name: Queue name to send message to.
        text: Text content to send.
        session_id: Optional session ID. If None, generates one.
        language: Language code (default: 'en').
        voice_id: Optional voice ID for TTS.
    """
    if session_id is None:
        session_id = f"demo-{datetime.now(timezone.utc).timestamp()}"

    message_data = {
        "type": "text",
        "data": text,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "language": language,
    }

    if voice_id:
        message_data["voice_id"] = voice_id

    await send_message(connection, queue_name, message_data)


async def send_multiple_messages(
    connection: aio_pika.Connection,
    queue_name: str,
    messages: List[str],
    delay: float = 2.0,
) -> None:
    """Send multiple messages with delay between them.

    Args:
        connection: RabbitMQ connection.
        queue_name: Queue name to send messages to.
        messages: List of text messages to send.
        delay: Delay between messages in seconds.
    """
    print(f"\nSending {len(messages)} messages to queue '{queue_name}'...")
    print(f"Delay between messages: {delay}s\n")

    for i, text in enumerate(messages, 1):
        session_id = f"demo-{i}-{datetime.now(timezone.utc).timestamp()}"
        await send_text_message(connection, queue_name, text, session_id=session_id)

        if i < len(messages):
            print(f"Waiting {delay}s before next message...\n")
            await asyncio.sleep(delay)

    print(f"\n✓ All {len(messages)} messages sent successfully!")


async def interactive_mode(connection: aio_pika.Connection, queue_name: str) -> None:
    """Interactive mode for sending messages.

    Args:
        connection: RabbitMQ connection.
        queue_name: Queue name to send messages to.
    """
    print("\n" + "=" * 60)
    print("Interactive Mode - Send messages to RabbitMQ")
    print("=" * 60)
    print(f"Queue: {queue_name}")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            text = input("Enter message text: ").strip()

            if not text:
                print("Empty message, skipping...")
                continue

            if text.lower() in ("quit", "exit", "q"):
                print("\nExiting interactive mode...")
                break

            await send_text_message(connection, queue_name, text)
            print()

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


async def send_from_file(
    connection: aio_pika.Connection,
    queue_name: str,
    file_path: Path,
) -> None:
    """Send messages from a JSON file.

    Args:
        connection: RabbitMQ connection.
        queue_name: Queue name to send messages to.
        file_path: Path to JSON file containing messages.
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    messages = []
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        if "messages" in data:
            messages = data["messages"]
        else:
            # Single message
            messages = [data]

    print(f"\nLoading {len(messages)} messages from {file_path}...")

    for msg_data in messages:
        if isinstance(msg_data, str):
            # Simple string message
            await send_text_message(connection, queue_name, msg_data)
        elif isinstance(msg_data, dict):
            # Full message object
            await send_message(connection, queue_name, msg_data)
        else:
            print(f"Warning: Skipping invalid message format: {msg_data}")

    print(f"\n✓ All messages from file sent successfully!")


def get_default_messages() -> List[str]:
    """Get default test messages.

    Returns:
        List of default test messages.
    """
    return [
        "Hello! This is a test message for the talking face PoC.",
        "The system should convert this text to speech and generate a talking face video.",
        "This is the third test message to verify sequential processing.",
        "All messages should be processed one after another.",
        "Thank you for testing the PoC system!",
    ]


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Demo script to send test messages to RabbitMQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send default test messages
  python scripts/demo_poc.py

  # Send a custom message
  python scripts/demo_poc.py --message "Hello, world!"

  # Interactive mode
  python scripts/demo_poc.py --interactive

  # Send messages from file
  python scripts/demo_poc.py --file messages.json

  # Send multiple custom messages
  python scripts/demo_poc.py --messages "First" "Second" "Third"
        """,
    )

    parser.add_argument(
        "--message",
        "-m",
        type=str,
        help="Single message text to send",
    )

    parser.add_argument(
        "--messages",
        nargs="+",
        help="Multiple messages to send",
    )

    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="JSON file containing messages to send",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode for sending messages",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between messages in seconds (default: 2.0)",
    )

    parser.add_argument(
        "--queue",
        type=str,
        help="RabbitMQ queue name (overrides config)",
    )

    parser.add_argument(
        "--host",
        type=str,
        help="RabbitMQ host (overrides config)",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="RabbitMQ port (overrides config)",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to environment file (default: .env or ENV_FILE env var)",
    )

    args = parser.parse_args()

    # Set dummy RTMP_URL if not set (demo script doesn't need it, but Settings requires it)
    import os
    if "RTMP_URL" not in os.environ:
        os.environ["RTMP_URL"] = "rtmp://dummy.test/live/demo"

    # Load settings with optional env_file
    settings = get_settings(env_file=args.env_file)
    rabbitmq_settings = settings.rabbitmq

    # Override with command-line arguments
    host = args.host or rabbitmq_settings.host
    port = args.port or rabbitmq_settings.port
    queue_name = args.queue or rabbitmq_settings.queue

    print("=" * 60)
    print("PoC Demo Script - RabbitMQ Message Sender")
    print("=" * 60)
    print(f"RabbitMQ: {host}:{port}")
    print(f"Queue: {queue_name}")
    print("=" * 60)

    # Connect to RabbitMQ
    try:
        connection_url = f"amqp://{rabbitmq_settings.user}:{rabbitmq_settings.password}@{host}:{port}/{rabbitmq_settings.vhost}"
        print(f"\nConnecting to RabbitMQ...")
        connection = await aio_pika.connect_robust(connection_url)
        print("✓ Connected to RabbitMQ\n")
    except Exception as e:
        print(f"✗ Failed to connect to RabbitMQ: {e}")
        print(f"\nMake sure RabbitMQ is running on {host}:{port}")
        sys.exit(1)

    try:
        # Determine mode
        if args.interactive:
            await interactive_mode(connection, queue_name)
        elif args.file:
            await send_from_file(connection, queue_name, args.file)
        elif args.message:
            await send_text_message(connection, queue_name, args.message)
        elif args.messages:
            await send_multiple_messages(
                connection, queue_name, args.messages, delay=args.delay
            )
        else:
            # Default: send default test messages
            messages = get_default_messages()
            await send_multiple_messages(
                connection, queue_name, messages, delay=args.delay
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await connection.close()
        print("\n✓ Connection closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)

