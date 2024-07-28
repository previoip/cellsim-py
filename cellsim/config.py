import os
import configparser
from collections import namedtuple
from re import compile as re_compile
from dataclasses import dataclass, asdict, fields

DEFAULT_CONTENT_SIZE = 1_000_000

regexp_hashinfo = re_compile(r'(?P<alg>\w+)\s*\((?P<f>.+)\)\s*\=\s*(?P<hash>\w+)\Z')
regexp_comment = re_compile(r'(?P<ws>.*)(?P<cc>[\#\;]+.*)')

@dataclass
class ConfigSectionSimDefault:
  request_per_step: int = 200
  seed: int = 2024
  dataset: str = 'ml-100k'
  results_dir: str = './results'

  @property
  def _section_name(self):
    return 'sim.default'

  @property
  def results_subdir(self):
    return '{}/{}'.format(self.results_dir, self.dataset)

@dataclass
class ConfigSectionSimCache:
  type: str = 'fifo'
  maxsize: float = 1000 * DEFAULT_CONTENT_SIZE
  maxage: float = 1000

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
  bs_topo: str = 'hex'
  num_clusters: int = 5
  ues_distribution: str = 'uniform'
  dist_min: float = 0.01
  dist_max: float = 0.3
  cell_radius: float = 0.3
  a: float = 3.5
  sigma2: float = 104.0
  shad_loss: float = 10
  rayleigh_scale: float = 1.0
  power_min: float = -10
  power_max: float = 20
  power_level: int = 10
  content_size: float = DEFAULT_CONTENT_SIZE
  bandwidth: float = 10e6
  assume_static: str = 'true'

  @property
  def _section_name(self):
    return 'sim.env'


@dataclass
class ConfigSectionSimRl:
  batch_size: int = 64
  gamma: float = 0.99
  eps_start: float = 0.9
  eps_end: float = 0.05
  eps_decay: float = 1000
  n_episode: int = 3
  tau: float = 0.005
  lr: float = 1e-4
  replay_temp_size: int = 1000

  @property
  def _section_name(self):
    return 'sim.rl'


@dataclass
class ConfigSectionDataset:
  download_dir: str = './downloads'
  extract_dir: str = './data'
  data_dir: str = './data'
  data_csv_requests: str = 'requests.csv'
  data_csv_inter: str = 'inter.csv'
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
    h = [t(*regexp_hashinfo.match(s).groups()) for s in self.archive_hash.splitlines() if regexp_hashinfo.match(s)]
    r = {i.file: i for i in h}
    return r


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
    self._parser = configparser.ConfigParser(inline_comment_prefixes=(';',))
    self._namespaces = [
      self.default,
      self.cache,
      self.recsys,
      self.env,
      self.rl,
      self.dataset_settings,
      self.dataset_info,
    ]
    self._comments = dict()

  @property
  def dataset(self) -> ConfigSectionDatasetMovielensVariant:
    return self.dataset_variants[self.default.dataset]

  def save(self):
    for namespace in self._namespaces:
      self._parser[namespace._section_name] = asdict(namespace)

    for dataset_name, dataset_namespace in self.dataset_variants.items():
      dataset_section = self.dataset_info._section_name + '.' + dataset_name
      self._parser[dataset_section] = asdict(dataset_namespace)

    _temp = list()
    with open(self._file_path, 'r+') as fp:
      self._parser.write(fp)
      # save inline comments
      fp.seek(0)
      for line in fp.readlines():
        _temp.append(line)
      fp.seek(0)
      for n, (offset, comment) in self._comments.items():
        if offset == 0:
          continue
        _temp[n] = _temp[n].rstrip()
        _temp[n] += (offset - len(_temp[n])) * ' '
        _temp[n] += comment
        _temp[n] += '\n'
      fp.writelines(_temp)

  def load(self):
    if not os.path.exists(self._file_path):
      self.save()

    with open(self._file_path, 'r') as fp:
      self._parser.read_file(fp)
      # load inline comments
      fp.seek(0)
      for n, line in enumerate(fp.readlines()):
        match_comment = regexp_comment.match(line)
        if not match_comment:
          continue
        self._comments[n] = (len(match_comment.group('ws')), match_comment.group('cc'))

    for namespace in self._namespaces:
      for field in fields(namespace):
        if not field.name in self._parser[namespace._section_name]:
          continue
        setattr(namespace, field.name, field.type(self._parser[namespace._section_name][field.name]))

    for dataset_name in self.dataset_info.variants_list:
      dataset_section = self.dataset_info._section_name + '.' + dataset_name
      namespace = ConfigSectionDatasetMovielensVariant()
      if not dataset_section in self._parser:
        continue
      for field in fields(namespace):
        if not field.name in self._parser[namespace._section_name]:
          continue
        setattr(namespace, field.name, field.type(self._parser[dataset_section][field.name]))
      self.dataset_variants[dataset_name] = namespace


def fmt(config: Config):
  return '{}_{}_{}_{}{}x{}_{}'.format(
    config.default.request_per_step,
    config.recsys.type,
    config.env.ues_distribution,
    config.env.bs_topo,
    config.env.bs_nx,
    config.env.bs_ny,
    config.cache.type
  )