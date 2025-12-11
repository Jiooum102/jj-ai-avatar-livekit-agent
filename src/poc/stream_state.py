"""Stream state management for the talking face livestream pipeline.

This module provides thread-safe state management for the stream pipeline,
tracking the current state (IDLE, PROCESSING, TALKING, TRANSITIONING) and
managing state transitions with callbacks.
"""

import logging
import threading
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream state enumeration."""

    IDLE = "idle"  # Streaming static/default video (no talking)
    PROCESSING = "processing"  # Generating talking face from RabbitMQ message
    TALKING = "talking"  # Streaming generated talking face video
    TRANSITIONING = "transitioning"  # Switching between static and talking video
    ERROR = "error"  # Error state


class StreamStateManager:
    """Thread-safe stream state manager with callbacks.

    This manager tracks the current stream state and provides callbacks
    for state changes. It ensures thread-safe state transitions and
    validates state change sequences.

    Example:
        ```python
        state_mgr = StreamStateManager()

        def on_state_change(old_state, new_state):
            print(f"State changed: {old_state} -> {new_state}")

        state_mgr.add_callback(on_state_change)
        state_mgr.set_state(StreamState.PROCESSING)
        ```
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        StreamState.IDLE: [StreamState.PROCESSING, StreamState.ERROR],
        StreamState.PROCESSING: [StreamState.TRANSITIONING, StreamState.IDLE, StreamState.ERROR],
        StreamState.TRANSITIONING: [StreamState.TALKING, StreamState.IDLE, StreamState.ERROR],
        StreamState.TALKING: [StreamState.TRANSITIONING, StreamState.IDLE, StreamState.ERROR],
        StreamState.ERROR: [StreamState.IDLE],  # Can recover from error to IDLE
    }

    def __init__(self, initial_state: StreamState = StreamState.IDLE) -> None:
        """Initialize stream state manager.

        Args:
            initial_state: Initial state. Defaults to StreamState.IDLE.
        """
        self._state = initial_state
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._callbacks: list[Callable[[StreamState, StreamState], None]] = []
        self._callback_lock = threading.Lock()

    @property
    def state(self) -> StreamState:
        """Get current state (thread-safe).

        Returns:
            Current stream state.
        """
        with self._lock:
            return self._state

    def set_state(self, new_state: StreamState, force: bool = False) -> bool:
        """Set new state with validation.

        Args:
            new_state: New state to transition to.
            force: If True, bypass transition validation. Defaults to False.

        Returns:
            True if state was changed, False if transition is invalid.

        Raises:
            ValueError: If transition is invalid and force=False.
        """
        with self._lock:
            old_state = self._state

            if old_state == new_state:
                return False  # No change needed

            # Validate transition
            if not force and not self._is_valid_transition(old_state, new_state):
                error_msg = f"Invalid state transition: {old_state} -> {new_state}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Update state
            self._state = new_state
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")

            # Notify callbacks
            self._notify_callbacks(old_state, new_state)

            return True

    def _is_valid_transition(self, from_state: StreamState, to_state: StreamState) -> bool:
        """Check if state transition is valid.

        Args:
            from_state: Current state.
            to_state: Target state.

        Returns:
            True if transition is valid, False otherwise.
        """
        valid_targets = self.VALID_TRANSITIONS.get(from_state, [])
        return to_state in valid_targets

    def add_callback(self, callback: Callable[[StreamState, StreamState], None]) -> None:
        """Add a callback for state changes.

        Args:
            callback: Callback function that takes (old_state, new_state) as arguments.
        """
        with self._callback_lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[StreamState, StreamState], None]) -> None:
        """Remove a callback.

        Args:
            callback: Callback function to remove.
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify_callbacks(self, old_state: StreamState, new_state: StreamState) -> None:
        """Notify all callbacks of state change.

        Args:
            old_state: Previous state.
            new_state: New state.
        """
        with self._callback_lock:
            callbacks = list(self._callbacks)  # Copy to avoid lock issues

        for callback in callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    def is_idle(self) -> bool:
        """Check if state is IDLE.

        Returns:
            True if state is IDLE, False otherwise.
        """
        return self.state == StreamState.IDLE

    def is_processing(self) -> bool:
        """Check if state is PROCESSING.

        Returns:
            True if state is PROCESSING, False otherwise.
        """
        return self.state == StreamState.PROCESSING

    def is_talking(self) -> bool:
        """Check if state is TALKING.

        Returns:
            True if state is TALKING, False otherwise.
        """
        return self.state == StreamState.TALKING

    def is_transitioning(self) -> bool:
        """Check if state is TRANSITIONING.

        Returns:
            True if state is TRANSITIONING, False otherwise.
        """
        return self.state == StreamState.TRANSITIONING

    def is_error(self) -> bool:
        """Check if state is ERROR.

        Returns:
            True if state is ERROR, False otherwise.
        """
        return self.state == StreamState.ERROR

    def reset_to_idle(self) -> bool:
        """Reset state to IDLE (can be called from any state).

        Returns:
            True if state was changed, False if already IDLE.
        """
        return self.set_state(StreamState.IDLE, force=True)

