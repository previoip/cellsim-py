import typing as t

class Node:
  def __init__(self, parent=None):
    self._parent: "Node" = parent
    self._children: t.List["Node"] = list()

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, o: "Node"):
    is_class = isinstance(o, self.__class__)
    is_subclass = object in self.__class__.__bases__ and isinstance(o, self.__class__.__bases__)
    if is_class or is_subclass:
      self._parent = o
    raise ValueError('object is not instance of {}'.format(self.__class__.__name__))

  @property
  def children(self):
    return self._children

  @property
  def rank(self):
    return len(self._children)

  @property
  def depth(self):
    n = 0
    for p in self.iter_parent():
      n += 1
    return n

  def is_leaf(self):
    return self.rank == 0
  
  def is_root(self):
    return self.parent is None

  def iter_parent(self):
    node = self.parent
    while not node is None:
      yield node
      node = node.parent

  def iter_child(self):
    yield self
    for child in self.children:
      yield from child.iter_child()

  def atindex(self):
    if self.is_root():
      return None
    return self.parent.children.index(self)

  def add(self, o: "Node"):
    self._children.append(o)
    o.parent = self

  def remove(self, o: "Node"):
    self._children.remove(o)
    o.parent = None