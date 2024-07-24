from threading import RLock

class TimestepTimer:
  _lock = RLock()
  def __init__(self):
    self._t0 = 0
    self._t1 = 0
    self._tt = 0

  def update(self, t):
    with self._lock:
      if self._t0 == 0:
        self._t0 = t
        self._t1 = t
        self._tt = t
        return
      if self._tt == t:
        return
      self._t1 = self._tt
      self._tt = t

  def delta_time(self):
    with self._lock:
      return self._tt - self._t1

  def elapsed(self):
    with self._lock:
      return self._tt - self._t0

  def reset(self):
    with self._lock:
      self._t0 = 0
      self._t1 = 0
      self._tt = 0
