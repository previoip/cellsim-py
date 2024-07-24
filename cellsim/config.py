import os
import configparser
from collections import namedtuple
from re import compile as re_compile
from dataclasses import dataclass, asdict, fields

regxp_hashinfo = re_compile(r'(?P<alg>\w+)\s*\((?P<f>.+)\)\s*\=\s*(?P<hash>\w+)\Z')

@dataclass
class ConfigSectionDataset:
  download_dir: str = './downloads'
  extract_dir: str = './data'
  data_dir: str = './data'
  variants: str = 'movielens'

  @property
  def _section_name(self):
    return 'dataset'

  @property
  def variants_list(self):
    return self.variants.splitlines()


@dataclass
class ConfigSectionDatasetMovielens:
  name: str = 'movielens'
  host: str = 'http://files.grouplens.org'
  path: str = 'datasets/movielens'
  variants: str = 'ml-100k\nml-1m\nml-10m\nml-20m\nml-25m\nml-32m'

  @property
  def _section_name(self):
    return 'dataset.movielens'

  @property
  def variants_list(self):
    return self.variants.splitlines()

@dataclass
class ConfigSectionDatasetMovielensVariant:
  archive_name: str = 'ml-100k.zip'
  archive_hash: str = 'MD5 (ml-100k.zip) = 0e33842e24a9c977be4e0107933c0723'
  data_file: str = 'u.data'
  delimiter: str = '\t'
  colname_uid: str = 'userId'
  colname_iid: str = 'movieId'
  colname_ts: str = 'timestamp'
  colname_inter: str = 'rating'
  colsequence: str = 'uid|iid|inter|ts'
  use_colnames_as_headers: str = 'true'

  @property
  def _section_name(self):
    return 'dataset.movielens.' + self.archive_name.split('.')[0]

  @property
  def hashinfo(self):
    t = namedtuple('HashInfo', ['alg', 'file', 'hash'])
    h = [t(*regxp_hashinfo.match(s).groups()) for s in self.archive_hash.splitlines() if regxp_hashinfo.match(s)]
    r = {i.file: i for i in h}
    return r

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
    self.dataset_settings = ConfigSectionDataset()
    self.dataset_info = ConfigSectionDatasetMovielens()
    self.dataset_variants = dict()
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
      self.dataset_settings,
      self.dataset_info,
    ]

  @property
  def dataset(self) -> ConfigSectionDatasetMovielensVariant:
    return self.dataset_variants[self.default.dataset]

  def save(self):
    for namespace in self._namespaces:
      self._parser[namespace._section_name] = asdict(namespace)

    for dataset_name, dataset_namespace in self.dataset_variants.items():
      dataset_section = self.dataset_info._section_name + '.' + dataset_name
      self._parser[dataset_section] = asdict(dataset_namespace)

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

    for dataset_name in self.dataset_info.variants_list:
      dataset_section = self.dataset_info._section_name + '.' + dataset_name
      namespace = ConfigSectionDatasetMovielensVariant()
      if not dataset_section in self._parser:
        continue
      for field in fields(namespace):
        setattr(namespace, field.name, field.type(self._parser[dataset_section][field.name]))
      self.dataset_variants[dataset_name] = namespace