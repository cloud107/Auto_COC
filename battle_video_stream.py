"""Continuously decode the emulator display and hand out only the latest frame."""

import os
import socket
import subprocess
import threading
import time

import cv2

from dragon_ice_block import VIEWPORT


class BattleVideoStream:
    def __init__(self, adb):
        self.adb = adb
        self.ready = threading.Condition()
        self.stopping = threading.Event()
        self.latest = None
        self.failure = None
        self.remote_pid = None

    def __enter__(self):
        self.listener = socket.socket()
        try:
            self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(1)
            self.listener.settimeout(5)
            width, height = VIEWPORT
            # The PID header belongs to the shell that execs screenrecord. Strip it before decoding.
            record = (f"echo $$; exec screenrecord --size {width}x{height} --bit-rate 6M "
                      "--time-limit 180 --output-format=h264 -")
            command = [*self.adb, "exec-out", record]
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError:
            self.listener.close()
            raise
        self.threads = [
            threading.Thread(target=self._bridge, daemon=True),
            threading.Thread(target=self._decode, args=(self.listener.getsockname()[1],), daemon=True),
        ]
        for thread in self.threads:
            thread.start()
        return self

    def _fail(self, message):
        with self.ready:
            if self.failure is None and not self.stopping.is_set():
                self.failure = message
            self.ready.notify_all()

    def _bridge(self):
        try:
            connection, _ = self.listener.accept()
            with connection:
                connection.settimeout(3)
                header = self.process.stdout.readline(32).strip()
                if not header.isdigit():
                    self._fail("MuMu did not report the video encoder PID")
                    return
                self.remote_pid = int(header)
                while not self.stopping.is_set():
                    packet = self.process.stdout.read1(65536)
                    if not packet:
                        break
                    connection.sendall(packet)
        except OSError as exc:
            self._fail(str(exc))

    def _decode(self, port):
        key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        previous = os.environ.get(key)
        os.environ[key] = "f;h264|probesize;4096|analyzeduration;0|fflags;nobuffer|flags;low_delay"
        capture = None
        try:
            try:
                capture = cv2.VideoCapture(
                    f"tcp://127.0.0.1:{port}", cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4500, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000],
                )
            finally:
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
            while not self.stopping.is_set():
                ok, frame = capture.read()
                if not ok:
                    self._fail("MuMu stopped sending video frames")
                    break
                if frame.shape[:2] != VIEWPORT[::-1]:
                    self._fail(f"Expected a {VIEWPORT[0]}x{VIEWPORT[1]} video frame")
                    break
                with self.ready:
                    self.latest = time.monotonic(), frame
                    self.ready.notify()
        except (OSError, cv2.error) as exc:
            self._fail(str(exc))
        finally:
            if capture is not None:
                capture.release()

    def read(self, earliest, *, timeout=2):
        """Wait for a fresh decoded frame at or beyond the requested sample time."""
        with self.ready:
            self.ready.wait_for(
                lambda: self.failure or self.stopping.is_set()
                or (self.latest is not None and self.latest[0] >= earliest),
                timeout=timeout,
            )
            if self.failure:
                raise OSError(self.failure)
            if self.latest is None or self.latest[0] < earliest or self.stopping.is_set():
                raise TimeoutError("No fresh MuMu video frame")
            return self.latest

    def __exit__(self, *_):
        self.stopping.set()
        self.listener.close()
        with self.ready:
            self.ready.notify_all()
        if self.remote_pid is not None and self.process.poll() is None:
            pid = self.remote_pid
            stop = f'if [ "$(cat /proc/{pid}/comm 2>/dev/null)" = screenrecord ]; then kill -INT {pid}; fi'
            try:
                subprocess.run([*self.adb, "shell", stop], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for thread in self.threads:
            thread.join(timeout=3)
        self.process.stdout.close()
