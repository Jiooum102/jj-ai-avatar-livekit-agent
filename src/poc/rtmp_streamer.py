"""RTMP streaming implementation using FFmpeg with FIFOs.

This module provides RTMP streaming functionality using FFmpeg with separate
FIFOs (named pipes) for video and audio inputs. It uses queues and writer
threads to decouple frame/audio generation from FFmpeg encoding.
"""

import errno
import logging
import os
import select
import subprocess
import sys
import threading
import time
import traceback
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np

from src.config.settings import RTMPSettings

logger = logging.getLogger(__name__)


class RTMPStreamer:
    """RTMP streamer using FFmpeg with FIFOs for video and audio."""

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        fps: int,
        *,
        video_pix_fmt: str = "rgb24",
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        audio_bitrate: str = "128k",
        video_bitrate: Optional[str] = None,
        audio_sample_rate: int = 16000,
        preset: str = "ultrafast",
        tune: Optional[str] = "zerolatency",
        gop_size: Optional[int] = None,
        data_dir: Optional[str] = None,
    ) -> None:
        """Initialize RTMP streaming with separate FIFOs for raw video and raw audio.

        Args:
            url: RTMP server URL (full URL including stream key).
            width: Video frame width.
            height: Video frame height.
            fps: Frames per second for the video stream.
            video_pix_fmt: Video pixel format. Defaults to "rgb24".
            video_codec: Video codec. Defaults to "libx264".
            audio_codec: Audio codec. Defaults to "aac".
            audio_bitrate: Audio bitrate. Defaults to "128k".
            video_bitrate: Video bitrate. If None, uses default.
            audio_sample_rate: Audio sample rate in Hz. Defaults to 16000.
            preset: FFmpeg preset. Defaults to "ultrafast".
            tune: FFmpeg tune option. Defaults to "zerolatency".
            gop_size: GOP (Group of Pictures) size. If None, uses fps.
            data_dir: Directory for FIFO files. Defaults to "tmp".
        """
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.video_pipe = None
        self.audio_pipe = None
        self.process = None

        # Encoding configuration
        self.video_pix_fmt = video_pix_fmt
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.video_bitrate = video_bitrate
        self.preset = preset
        self.tune = tune
        self.gop_size = gop_size

        # Audio config (mono PCM s16 from pipeline)
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = 1

        # FIFO-based streaming (unique paths per process/thread)
        if data_dir is None:
            data_dir = "tmp"
        os.makedirs(data_dir, exist_ok=True)
        self.video_fifo_path = f"{data_dir}/rtmp_video_fifo_{os.getpid()}_{threading.get_ident()}"
        self.audio_fifo_path = f"{data_dir}/rtmp_audio_fifo_{os.getpid()}_{threading.get_ident()}"
        self.video_fifo_fp = None  # file object used to write frames
        self.audio_fifo_fp = None  # file object used to write audio samples

        # Queues and writer threads for decoupled A/V writing
        buffer_frames = max(2, self.fps * 4)
        self.video_queue: Queue = Queue(maxsize=buffer_frames)
        self.audio_queue: Queue = Queue(maxsize=buffer_frames)
        self._stop_event = threading.Event()
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._dropped_video = 0
        self._dropped_audio = 0

        # OS FIFO pipe buffer target size (bytes)
        self.fifo_buffer_size = 1 << 20  # 1 MiB default
        self.fifo_buffer_size = self.fifo_buffer_size * 100  # 100 MiB

        self.start_time = None
        self.frame_count = 0

    def start(self) -> None:
        """Start the FFmpeg process for RTMP streaming (using FIFOs instead of stdin pipes)."""
        # Create FIFOs
        try:
            if os.path.exists(self.video_fifo_path):
                os.remove(self.video_fifo_path)
            if os.path.exists(self.audio_fifo_path):
                os.remove(self.audio_fifo_path)
            os.mkfifo(self.video_fifo_path)
            os.mkfifo(self.audio_fifo_path)
        except Exception as e:
            logger.error(f"Failed to create FIFOs at {self.video_fifo_path}, {self.audio_fifo_path}: {e}")
            raise

        # Compute GOP if not provided
        gop = self.gop_size if self.gop_size is not None else max(int(round(self.fps)), 1)

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-thread_queue_size",
            "1024",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.video_pix_fmt,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps}",
            "-i",
            self.video_fifo_path,
            "-thread_queue_size",
            "1024",
            "-f",
            "s16le",
            "-ac",
            str(self.audio_channels),
            "-ar",
            str(self.audio_sample_rate),
            "-i",
            self.audio_fifo_path,
            "-c:v",
            self.video_codec,
            "-preset",
            self.preset,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-c:a",
            self.audio_codec,
            "-b:a",
            self.audio_bitrate,
        ]
        if self.tune:
            cmd.extend(["-tune", self.tune])
        if self.video_bitrate:
            cmd.extend(["-b:v", self.video_bitrate])
        # Output to RTMP
        cmd.extend(["-f", "flv", self.url])

        try:
            self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        except Exception as e:
            logger.error(f"Failed to start FFmpeg RTMP process: {e}\n{traceback.format_exc()}")
            # Cleanup FIFOs if ffmpeg failed to start
            self.stop()
            raise

        # Open FIFOs for writing. Use O_RDWR to avoid blocking if FFmpeg hasn't opened the reader yet.
        try:
            vfd = os.open(self.video_fifo_path, os.O_RDWR)
            self.video_fifo_fp = os.fdopen(vfd, "wb", buffering=0)
            afd = os.open(self.audio_fifo_path, os.O_RDWR)
            self.audio_fifo_fp = os.fdopen(afd, "wb", buffering=0)
        except Exception as e:
            logger.error(f"Failed to open FIFOs for writing: {e}")
            self.stop()
            raise

        # Start writer threads
        self._stop_event.clear()
        self._video_thread = threading.Thread(target=self._video_writer_loop, name="video-writer", daemon=True)
        self._audio_thread = threading.Thread(target=self._audio_writer_loop, name="audio-writer", daemon=True)
        self._video_thread.start()
        self._audio_thread.start()

        logger.info(f"RTMP streamer started: {self.url} ({self.width}x{self.height}@{self.fps}fps)")

    def _write_with_timeout(self, fd: int, data: bytes, timeout_sec: float, label: str) -> bool:
        """Write data to a non-blocking FIFO FD with timeout.

        Returns True if all bytes written, else False.

        Args:
            fd: File descriptor to write to.
            data: Data bytes to write.
            timeout_sec: Timeout in seconds.
            label: Label for logging (e.g., "video" or "audio").

        Returns:
            True if all bytes written, False otherwise.
        """
        total = 0
        nbytes = len(data)
        end_time = time.time() + timeout_sec
        while total < nbytes:
            now = time.time()
            if now >= end_time:
                logger.warning(f"Timeout writing to {label} FIFO: wrote {total}/{nbytes} bytes. Dropping remainder.")
                return False
            # Wait until writable
            _, wlist, _ = select.select([], [fd], [], max(0.0, end_time - now))
            if not wlist:
                continue
            try:
                written = os.write(fd, data[total:])
                if written == 0:
                    logger.error(f"Write returned 0 for {label} FIFO. Reader closed?")
                    return False
                total += written
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR):
                    continue
                logger.error(f"OS error writing to {label} FIFO: {e}")
                return False
        return True

    def _video_writer_loop(self) -> None:
        """Writer thread loop for video frames."""
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.video_queue.get(timeout=0.1)
                except Empty:
                    continue
                if frame is None:
                    break
                if self.video_fifo_fp is None:
                    break
                try:
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    frame = np.ascontiguousarray(frame)
                    video_bytes = frame.tobytes()
                    ok = self._write_with_timeout(
                        self.video_fifo_fp.fileno(), video_bytes, timeout_sec=30, label="video"
                    )
                    if not ok:
                        logger.warning("Video write timed out; dropped frame.")
                        self._dropped_video += 1
                except Exception as e:
                    logger.error(f"Video writer error: {e}. Traceback: {traceback.format_exc()}")
                    self._stop_event.set()
                    break
        except Exception as e:
            logger.error(f"Video writer thread crashed: {e}")
            self._stop_event.set()
        logger.info("Video writer thread exiting.")

    def _audio_writer_loop(self) -> None:
        """Writer thread loop for audio chunks."""
        try:
            while not self._stop_event.is_set():
                try:
                    audio = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                if audio is None:
                    break
                if self.audio_fifo_fp is None:
                    break
                try:
                    a = np.asarray(audio) if audio is not None else None
                    if a is None or a.size == 0:
                        # nothing to write for this tick
                        continue
                    if a.ndim > 1:
                        a = a.mean(axis=1)
                    if a.dtype != np.int16:
                        a = a.astype(np.float32)
                        a = np.clip(a, -1.0, 1.0)
                        a = (a * 32767.0).astype(np.int16)
                    audio_bytes = a.tobytes()
                    ok = self._write_with_timeout(
                        self.audio_fifo_fp.fileno(), audio_bytes, timeout_sec=30, label="audio"
                    )
                    if not ok:
                        logger.warning("Audio write timed out; dropped chunk.")
                        self._dropped_audio += 1

                except Exception as e:
                    logger.error(f"Audio writer error: {e}. Traceback: {traceback.format_exc()}")
                    self._stop_event.set()
                    break
        except Exception as e:
            logger.error(f"Audio writer thread crashed: {e}")
            self._stop_event.set()
        logger.info("Audio writer thread exiting.")

    def push(self, frame: np.ndarray, audio: np.ndarray) -> None:
        """Enqueue a video frame (RGB np.uint8) and a mono audio chunk (int16 or float in [-1,1]).

        Writer threads will process and push to FIFOs.

        Args:
            frame: Video frame as numpy array (RGB, uint8).
            audio: Audio chunk as numpy array (mono, int16 or float in [-1,1]).

        Raises:
            RuntimeError: If FFmpeg process is not running.
        """
        # Check if streaming is active
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("FFmpeg process is not initialized or has exited.")

        if self.start_time is None:
            self.start_time = time.time()

        self.frame_count += 1
        while time.time() - self.start_time < self.frame_count / self.fps:
            time.sleep(0.001)

        while not self._stop_event.is_set():
            try:
                self.video_queue.put(frame, timeout=0.1)
                break
            except Full:
                continue

        while not self._stop_event.is_set():
            try:
                self.audio_queue.put(audio, timeout=0.1)
                break
            except Full:
                continue

    def stop(self) -> None:
        """Stop the RTMP streaming process and clean up FIFOs."""
        # Signal writer threads to stop and unblock queues
        self._stop_event.set()
        try:
            self.video_queue.put_nowait(None)
        except Exception:
            pass
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass
        # Join writer threads
        try:
            if self._video_thread and self._video_thread.is_alive():
                self._video_thread.join(timeout=2)
        except Exception:
            pass
        finally:
            self._video_thread = None
        try:
            if self._audio_thread and self._audio_thread.is_alive():
                self._audio_thread.join(timeout=2)
        except Exception:
            pass
        finally:
            self._audio_thread = None

        # Close FIFO writers
        if self.video_fifo_fp:
            try:
                self.video_fifo_fp.close()
            except Exception:
                pass
            finally:
                self.video_fifo_fp = None
        if self.audio_fifo_fp:
            try:
                self.audio_fifo_fp.close()
            except Exception:
                pass
            finally:
                self.audio_fifo_fp = None

        # Terminate ffmpeg
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except Exception:
                    self.process.kill()
                    self.process.wait()
            except Exception:
                pass
            finally:
                try:
                    if getattr(self.process, "returncode", 0) not in (0, None):
                        stderr = (
                            self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
                        )
                        logger.error(f"ffmpeg exited with {self.process.returncode}: {stderr}")
                except Exception:
                    pass
                self.process = None

        # Remove FIFO files
        try:
            if self.video_fifo_path and os.path.exists(self.video_fifo_path):
                os.remove(self.video_fifo_path)
        except Exception as e:
            logger.warning(f"Failed to remove FIFO {self.video_fifo_path}: {e}")
        try:
            if self.audio_fifo_path and os.path.exists(self.audio_fifo_path):
                os.remove(self.audio_fifo_path)
        except Exception as e:
            logger.warning(f"Failed to remove FIFO {self.audio_fifo_path}: {e}")

        self.video_pipe = None
        self.audio_pipe = None

        logger.info("RTMP streamer stopped")


def create_rtmp_streamer(settings: RTMPSettings, data_dir: Optional[str] = None) -> RTMPStreamer:
    """Create an RTMP streamer from settings.

    Args:
        settings: RTMP settings configuration.
        data_dir: Directory for FIFO files. Defaults to "tmp".

    Returns:
        Configured RTMPStreamer instance.
    """
    return RTMPStreamer(
        url=settings.url,
        width=settings.width,
        height=settings.height,
        fps=settings.fps,
        audio_bitrate=settings.audio_bitrate,
        audio_sample_rate=settings.audio_sample_rate,
        video_bitrate=settings.bitrate,
        data_dir=data_dir,
    )

