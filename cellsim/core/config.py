import os
import configparser
from dataclasses import dataclass, asdict, fields


@dataclass
class ConfigSectionSimDefault:
  request_per_step: int = 200
  seed: int = 2024
  dataset: str = 'ml-100k'

  @property
  def _section_name(self):
    return 'sim.default'


@dataclass
class ConfigSectionSimCache:
  type: str = 'fifo'
  maxsize: int = 1000

  @property
  def _section_name(self):
    return 'sim.cache'


@dataclass
class ConfigSectionSimRecsys:
  type: str = 'mostpop'

  @property
  def _section_name(self):
    return 'sim.recsys'


@dataclass
class ConfigSectionSimEnv:
  bs_nx: int = 4
  bs_ny: int = 6
  num_clusters: int = 5
  min_dist: float = 0.01
  max_dist: float = 0.3
  cell_radius: float = 0.3
  a: float = 3.5
  sigma2: float = 104.0
  shad_loss: float = 10
  min_power: float = -10
  max_power: float = 20
  power_set_levels: int = 10
  content_size: float = 10e6
  ue_bandwidth: float = 10e6

  @property
  def _section_name(self):
    return 'sim.env'

@dataclass
class ConfigSectionSimRl:
  batch_size: float = 64
  gamma: float = 0.99
  eps_start: float = 0.9
  eps_end: float = 0.05
  eps_decay: float = 1000
  n_episode: float = 3
  tau: float = 0.005
  lr: float = 1e-4
  replay_buf_size: int = 1000

  @property
  def _section_name(self):
    return 'sim.rl'


class Config:

  def __init__(self, file_path):
    self._file_path = file_path
    self.default: ConfigSectionSimDefault = ConfigSectionSimDefault()
    self.cache: ConfigSectionSimCache = ConfigSectionSimCache()
    self.recsys: ConfigSectionSimRecsys = ConfigSectionSimRecsys()
    self.env: ConfigSectionSimEnv = ConfigSectionSimEnv()
    self.rl: ConfigSectionSimRl = ConfigSectionSimRl()
    self._parser = configparser.ConfigParser()
    self._namespaces = [
      self.default,
      self.cache,
      self.recsys,
      self.env,
      self.rl,
    ]

  def save(self):
    for namespace in self._namespaces:
      self._parser[namespace._section_name] = asdict(namespace)
    with open(self._file_path, 'w') as fp:
      self._parser.write(fp)

  def load(self):
    if not os.path.exists(self._file_path):
      self.save()
    with open(self._file_path, 'r') as fp:
      self._parser.read_file(fp)
    for namespace in self._namespaces:
      for field in fields(namespace):
        setattr(namespace, field.name, field.type(self._parser[namespace._section_name][field.name]))