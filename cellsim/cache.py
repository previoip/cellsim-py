from collections import OrderedDict, namedtuple
from cellsim.core.timer import TimeTimer

class BaseCache:
  t_cache_record = namedtuple('CacheRecord', field_names=['ttl', 'size', 'item'])

  def __init__(self, timer, maxsize=1000.0, maxage=-1):
    self.timer = timer
    self.maxsize = maxsize
    self.maxage = maxage
    self.cache = OrderedDict()
    self.tempbuf = list()

  def __contains__(self, key):
    return self.has(key)

  def __iter__(self):
    yield from self.cache.keys()

  @property
  def usage(self):
    return sum([i.size for i in self.cache.values() if i.size > 0])

  @property
  def space_left(self):
    return self.maxsize - self.usage

  @property
  def frac_usage(self):
    if self.maxsize == 0:
      return 1
    return self.usage / self.maxsize

  def is_full(self):
    return self.maxsize >= 0 and self.frac_usage > 1

  def clear(self):
    self.cache.clear()
    self.tempbuf.clear()

  def has(self, key):
    return not self.cache.get(key, None) is None

  def delete(self, key):
    del self.cache[key]

  def add(self, key, item, size=1, ttl=None):
      retval = 0
      if ttl is None:
        ttl = self.maxage
      ttl += self.timer.now()
      if self.has(key):
        self.cache.move_to_end(key)
        # rec = self.cache[key]
        # rec.ttl = ttl
        retval = 1
      else:
        self.cache[key] = self.t_cache_record(ttl, size, item)
      return retval


class FIFOCache(BaseCache):
  def evict(self):
    timestamp = self.timer.now()
    popped = list()
    for key, rec in self.cache.items():
      if rec.ttl != -1 and timestamp >= rec.ttl:
        popped.append(rec)
        self.delete(key)
    if self.is_full():
      sums = 0
      to_delete = list()
      space_left = self.space_left
      for k, i in sorted([(k, i) for k, i in self.cache.items()], key=lambda x: x[1].ttl, reverse=True):
        sums += i.size
        if sums > space_left:
          to_delete.append(k)
      for k in to_delete:
        rec = self.cache[k]
        popped.append(rec)
        self.delete(k)
    return popped


class LIFOCache(BaseCache):
  def evict(self):
    timestamp = self.timer.now()
    popped = list()
    for key, rec in self.cache.items():
      if rec.ttl != -1 and timestamp >= rec.ttl:
        popped.append(rec)
        self.delete(key)
    if self.is_full():
      sums = 0
      to_delete = list()
      space_left = self.space_left
      for k, i in sorted([(k, i) for k, i in self.cache.items()], key=lambda x: x[1].ttl, reverse=False):
        sums += i.size
        if sums > space_left:
          to_delete.append(k)
      for k in to_delete:
        rec = self.cache[k]
        popped.append(rec)
        self.delete(k)
    return popped
