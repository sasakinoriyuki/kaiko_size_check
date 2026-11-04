"""Local process locks; no stale PID files are treated as active locks."""
from pathlib import Path
import os


class FileLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open('a+b')
        if not self.path.stat().st_size:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        if not self.acquire():
            raise BlockingIOError('他の起動処理が動いています。')
        return self

    def __exit__(self, *_):
        self.release()
