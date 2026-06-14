import logging
import threading
import time
from collections import deque

from pythonjsonlogger.json import JsonFormatter


class EventLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_file, buffer_size=1000, flush_interval=1):
        self.log_file = log_file
        self.buffer = deque()
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.lock = threading.Lock()
        self._setup_logger()
        self._start_flush_thread()

    def _setup_logger(self):
        if hasattr(self, '_initialized') and self._initialized:
            return  # Avoid re-initializing the logger
        self.logger = logging.getLogger("market_events")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file)
        formatter = JsonFormatter('%(event_type)s %(actor_id)s %(entity_id)s %(other_id)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _start_flush_thread(self):
        thread = threading.Thread(target=self._flush_periodically, daemon=True)
        thread.start()

    def _flush_periodically(self):
        while True:
            time.sleep(self.flush_interval)
            self.flush()

    def log_event(self, event_type, actor_id=None, entity_id=None, other_id=None):
        event = {
            "event_type": event_type,
            "actor_id": actor_id,
            "entity_id": entity_id,
            "other_id": other_id
        }
        with self.lock:
            self.buffer.append(event)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def flush(self):
        with self.lock:
            while self.buffer:
                event = self.buffer.popleft()
                self.logger.info("", extra=event)

    def shutdown(self):
        self.flush()
        # Optionally, remove handlers or perform other cleanup
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
