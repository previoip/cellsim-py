from collections import OrderedDict, namedtuple
from cellsim.core.timer import TimeTimer

class BaseCache:
  t_cache_record = namedtuple('CacheRecord', field_names=['ttl', 'size', 'item'])

  def __init__(self, timer, maxsize=1000.0, maxage=-1):
    self.timer = timer
    self.maxsize = maxsize
    self.maxage = maxage
    self.cache = OrderedDict()

  def __contains__(self, key):
    return self.has(key)

  def __iter__(self):
    yield from self.cache.keys()

  @property
  def usage(self):
    return sum([i.size for i in self.cache.values() if i.size > 0])

  @property
  def frac_usage(self):
    if self.maxsize == 0:
      return 1
    return self.usage / self.maxsize

  def is_full(self):
    return self.maxsize >= 0 and self.frac_usage > 1

  def clear(self):
    self.cache.clear()

  def has(self, key):
    return not self.cache.get(key, None) is None

  def delete(self, key):
    del self.cache[key]

class FIFOCache(BaseCache):
  def add(self, key, item, size=1, ttl=None):
    retval = 0
    if ttl is None:
      ttl = self.maxage
    ttl += self.timer.now()
    if self.has(key):
      self.delete(key)
      retval = 1
    self.cache[key] = self.t_cache_record(ttl, size, item)
    return retval

  def evict(self):
    timestamp = self.timer.now()
    popped = list()
    for key, rec in self.cache.items():
      if rec.ttl != -1 and timestamp >= rec.ttl:
        popped.append(rec)
        self.delete(key)

    while self.is_full():
      key = next(iter(self))
      rec = self.cache[key]
      popped.append(rec)
      self.delete(key)

    return popped
