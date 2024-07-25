import time

class TimeTimer:
  def __init__(self, t_0=None):
    self.t_0 = time.time()
    self.t_p = self.t_0
    self.t_t = self.t_0

  def update(self, t=None):
    self.t_p = self.t_t
    self.t_t = time.time()

  def delta_time(self):
    return self.t_t - self.t_p

  def elapsed(self):
    return self.t_t - self.t_0

  def reset(self):
    self.t_t = time.time()
    self.t_p = self.t_t

  def now(self):
    return self.t_t


class TimestepTimer:
  def __init__(self, t_0=0):
    self.t_0 = t_0
    self.t_p = 0
    self.t_t = 0

  def update(self, t):
    self.t_p = self.t_t
    self.t_t = t

  def delta_time(self):
    return self.t_t - self.t_p

  def elapsed(self):
    return self.t_t - self.t_0

  def reset(self):
    self.t_t = 0
    self.t_p = 0

  def now(self):
    return self.t_t

