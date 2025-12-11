#!/usr/bin/env python3
"""Simple script to send a message to RabbitMQ queue.

This is a minimal example for sending text messages to RabbitMQ.

Usage:
    python scripts/send_rabbitmq_message.py "Hello, this is a test message"
    python scripts/send_rabbitmq_message.py --text "Hello, world!"
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import aio_pika

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings


async def send_message_to_rabbitmq(text: str, queue_name: str, connection_url: str) -> None:
    """Send a text message to RabbitMQ queue.
    
    Args:
        text: Text message to send.
        queue_name: RabbitMQ queue name.
        connection_url: RabbitMQ connection URL.
    """
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(connection_url)
    
    try:
        # Create a channel
        channel = await connection.channel()
        
        # Declare the queue (creates if doesn't exist)
        queue = await channel.declare_queue(queue_name, durable=True)
        
        # Create message data
        message_data = {
            "type": "text",
            "data": text,
            "session_id": f"simple-{datetime.now(timezone.utc).timestamp()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": "en",
        }
        
        # Convert to JSON and encode
        message_body = json.dumps(message_data).encode("utf-8")
        
        # Create message
        message = aio_pika.Message(
            message_body,
            content_type="application/json",
            timestamp=datetime.now(timezone.utc),
        )
        
        # Publish message to queue
        await channel.default_exchange.publish(message, routing_key=queue_name)
        
        print(f"✓ Message sent successfully!")
        print(f"  Queue: {queue_name}")
        print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Session ID: {message_data['session_id']}")
        
        # Close channel
        await channel.close()
        
    finally:
        # Close connection
        await connection.close()


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Simple script to send a message to RabbitMQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send a simple message
  python scripts/send_rabbitmq_message.py "Hello, world!"
  
  # Send with explicit --text flag
  python scripts/send_rabbitmq_message.py --text "This is a test message"
  
  # Use custom env file
  python scripts/send_rabbitmq_message.py --text "Hello" --env-file .env.dev
        """,
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        help="Text message to send",
    )
    
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        dest="text_flag",
        help="Text message to send (alternative to positional argument)",
    )
    
    parser.add_argument(
        "--queue",
        "-q",
        type=str,
        help="RabbitMQ queue name (overrides config)",
    )
    
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to environment file (default: .env or ENV_FILE env var)",
    )
    
    args = parser.parse_args()
    
    # Get message text (from positional argument or --text flag)
    text = args.text or args.text_flag
    if not text:
        parser.error("Please provide a text message (as positional argument or --text)")
    
    # Set dummy RTMP_URL if not set (script doesn't need it, but Settings requires it)
    import os
    if "RTMP_URL" not in os.environ:
        os.environ["RTMP_URL"] = "rtmp://dummy.test/live/demo"
    
    # Load settings
    settings = get_settings(env_file=args.env_file)
    rabbitmq_settings = settings.rabbitmq
    
    # Get queue name
    queue_name = args.queue or rabbitmq_settings.queue
    
    # Build connection URL
    connection_url = rabbitmq_settings.connection_url
    
    print("=" * 60)
    print("Send Message to RabbitMQ")
    print("=" * 60)
    print(f"RabbitMQ: {rabbitmq_settings.host}:{rabbitmq_settings.port}")
    print(f"Queue: {queue_name}")
    print("=" * 60)
    print()
    
    try:
        await send_message_to_rabbitmq(text, queue_name, connection_url)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)

