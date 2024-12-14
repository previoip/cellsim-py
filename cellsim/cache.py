from collections import OrderedDict, namedtuple
from cellsim.core.timer import TimeTimer


class BaseCache:
  t_cache_record = namedtuple('CacheRecord', field_names=['ttl', 'size', 'item'])

  def __init__(self, timer, maxsize=1000.0, maxage=-1):
    self.timer = timer
    self.maxsize = maxsize
    self.maxage = maxage
    self.freq = dict()
    self.cache = OrderedDict()

    self.sums_wr = 0
    self.sums_rd = 0
    self.sums_rm = 0

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

  def reset(self):
    self.clear()
    self.sums_wr = 0
    self.sums_rd = 0
    self.sums_rm = 0

  def has(self, key):
    return not self.cache.get(key, None) is None

  def move_to_end(self, key):
    self.cache.move_to_end(key)

  def delete(self, key):
    self.freq_del(key)
    del self.cache[key]

  def popitem(self, last):
    key, rec = self.cache.popitem(last=last)
    self.freq_del(key)
    return key, rec

  def add(self, key, item, size=1, ttl=None):
      if self.has(key):
        self.move_to_end(key)
        self.freq_incr(key)
        self.sums_rd += size
        return ('read', size)
      if ttl is None:
        ttl = self.maxage
      ttl += self.timer.now()
      self.cache[key] = self.t_cache_record(ttl, size, item)
      return ('write', size)

  def freq_assert(self, key):
    if not key in self.freq:
      self.freq[key] = 0

  def freq_incr(self, key):
    self.freq_assert(key)
    self.freq[key] += 1

  def freq_del(self, key):
    if key in self.freq and key in self.cache:
      # del self.freq[key]
      self.freq[key] = 0

  def freq_reset(self):
    for k in self.freq.keys():
      self.freq[k] = 0

  def set_hit(self, key):
    self.freq_incr(key)

  def p_evict_tlru(self):
    deleted_sums = 0
    if self.maxsize == 0:
      return
    while self.is_full():
      key, rec = self.popitem(last=False)
      self.freq_del(key)
      deleted_sums += rec.size
    return deleted_sums

  def p_evict_lfru(self):
    deleted_sums = 0
    while self.is_full():
      key = max(self.freq, key=self.freq.get)
      if not key in self.cache:
        self.freq.pop(key)
        continue
      rec = self.cache[key]
      deleted_sums += rec.size
      self.delete(key)
    return deleted_sums

  def p_evict_expired(self):
    timestamp = self.timer.now()
    deleted_sums = 0
    keys = list(self.cache.keys())
    for key in keys:
      rec = self.cache[key]
      if rec.ttl != -1 and timestamp >= rec.ttl:
        deleted_sums += rec.size
        self.delete(key)
    return deleted_sums

  def p_evict_timebound(self, fifo=True):
    deleted_sums = 0
    sums = 0
    to_delete = list()
    space_left = self.maxsize
    for k, i in sorted([(k, i) for k, i in self.cache.items()], key=lambda x: x[1].ttl, reverse=fifo):
      sums += i.size
      if sums > space_left:
        to_delete.append(k)
    for k in to_delete:
      rec = self.cache[k]
      deleted_sums += rec.size
      self.delete(k)
    return deleted_sums


class FIFOCache(BaseCache): # FIFO-LRU
  def evict(self):
    deleted_sums = 0
    if self.is_full():
      deleted_sums += self.p_evict_timebound(fifo=True)
    deleted_sums += self.p_evict_expired()
    return deleted_sums

class LIFOCache(BaseCache): # LIFO-LRU
  def evict(self):
    deleted_sums = 0
    if self.is_full():
      deleted_sums += self.p_evict_timebound(fifo=False)
    deleted_sums += self.p_evict_expired()
    return deleted_sums

class TLRUCache(BaseCache):
  def evict(self):
    deleted_sums = 0
    deleted_sums += self.p_evict_tlru()
    return deleted_sums

class LFRUCache(BaseCache):
  def evict(self):
    deleted_sums = 0
    deleted_sums += self.p_evict_lfru()
    return deleted_sums