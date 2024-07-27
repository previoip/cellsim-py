from collections import namedtuple, deque

class ReplayBuffer:
  Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

  def __init__(self, maxlen):
    self.mem = deque([], maxlen=maxlen)

  def add(self, item):
    self.mem.append(item)

  def sample(self, n=1):
    return random.sample(self.mem, k=n)

  def len(self):
    return len(self.mem)

  @staticmethod
  def gen_mask(samples, attr='next_state'):
    yield from map(lambda x: not getattr(x, attr) is None, samples)
