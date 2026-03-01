from __future__ import annotations
import threading
from typing import Optional

class BotController:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False

    def start(self, sleep_s: int):
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.orchestrator.run_forever,
            kwargs={"sleep_s": sleep_s, "stop_event": self._stop_event},
            daemon=True
        )
        self._thread.start()
        self.is_running = True

    def stop(self):
        if not self.is_running:
            return
        self._stop_event.set()
        self.is_running = False
