import numpy as np
import matplotlib.pyplot as plt
import tqdm
from cellsim.core.timer import TimestepTimer
from cellsim.cache import FIFOCache, LIFOCache
from cellsim.config import Config

SIN60 = np.sin(np.deg2rad(60))
ATOL = 1e-8

class RequestWrapper:
  def __init__(self, df_request, config: Config, c_uid='uid', c_iid='iid', c_ts='ts', c_inter='rating'):
    self.df = df_request
    self.c_uid = c_uid
    self.c_iid = c_iid
    self.c_ts = c_ts
    self.c_inter = c_inter
    self.uids = df_request[c_uid].drop_duplicates().sort_values().to_numpy()
    self.iids = df_request[c_iid].drop_duplicates().sort_values().to_numpy()
    self.tss = df_request[c_ts].drop_duplicates().sort_values().to_numpy()
    self.n_batch = config.default.request_per_step
    self.n_total = len(self.df)
    self.counter = 0
    self.stop_flag = False

  @property
  def cursor(self):
    return self.counter * self.n_batch

  def _it_cursor(self):
    self.counter += 1
    if self.cursor > self.n_total:
      self.stop_flag = True

  def get_next_batch(self):
    view = self.df[self.cursor:self.cursor+self.n_batch]
    self._it_cursor()
    return view

  def reset(self):
    self.counter = 0
    self.stop_flag = False


class CellNetEnviron:

  @staticmethod
  def topo_factory_grid(nx, ny):
    x = np.linspace(-.5, .5, nx)
    y = np.linspace(-.5, .5, ny)
    ax, ay = np.meshgrid(x, y)
    ax, ay = ax.flatten(), ay.flatten()
    return np.column_stack((ax, ay, np.zeros(ax.shape)))

  @staticmethod
  def topo_factory_hexlattice(nx, ny):
    x = np.linspace(-.5, .5, nx)
    y = np.linspace(-.5, .5, ny)
    ax, ay = np.meshgrid(x, y)
    ax[1::2] += (0.25 / (nx-1))
    ax[::2] -= (0.25 / (nx-1))
    ay *= SIN60
    ax, ay = ax.flatten(), ay.flatten()
    return np.column_stack((ax, ay, np.zeros(ax.shape)))

  def __init__(self, request_df: RequestWrapper, config: Config):
    # anything isn't prefrixed is either scalar or instances
    self.request_w = RequestWrapper(request_df, config)
    self.n_ue = len(self.request_w.uids)
    self.n_ii = len(self.request_w.iids)
    self.bs_nx = config.env.bs_nx
    self.bs_ny = config.env.bs_ny
    self.N = self.bs_nx * self.bs_ny
    self.L = config.env.cell_radius
    self.a = config.env.a
    self.sigma2 = config.env.sigma2
    self.rayleigh_scale = 1
    self.f_static = config.env.assume_static == 'true'
    self.timer = TimestepTimer()
    self.a_power_set = np.linspace(config.env.power_min, config.env.power_max, config.env.power_level)
    self.a_content_size = np.ones((self.n_ii)) * config.env.content_size
    self._do_update_dist = False

    # vectors
    #   *_pos: cart coord [x, y, z]
    #   *_hea: heading (antenna) [elev, azim]
    #   *_vel: cart velocity [u, v, w]
    print('allocating vectors')
    self.v_ues_pos = np.zeros((self.n_ue, 3))
    self.v_ues_hea = np.zeros((self.n_ue, 2))
    self.v_ues_vel = np.zeros((self.n_ue, 3))
    self.v_bss_pos = np.zeros((self.N, 3))
    self.v_bss_hea = np.zeros((self.N, 2))
    self.v_bss_vel = np.zeros((self.N, 3))

    # pre-alloc arrays/matrix
    #   *_d*: delta *
    #   *_dist : sqrt(square(*_dl))
    #   *_gr: group indices
    #   *_grm: group mask
    #   *_adj: adjacent mask
    print('allocating arrays/matrixes')
    self.m_bss2bss_dpos = np.zeros((self.N, self.N, 3))
    self.m_ues2ues_dpos = np.zeros((self.n_ue, self.n_ue, 3))
    self.m_ues2bss_dpos = np.zeros((self.n_ue, self.N, 3))
    self.m_bss2bss_dist = np.zeros((self.N, self.N))
    self.m_ues2ues_dist = np.zeros((self.n_ue, self.n_ue))
    self.m_ues2bss_dist = np.zeros((self.n_ue, self.N))
    self.a_ues_gr = np.zeros((self.n_ue), dtype=np.int64)
    self.m_bss2ues_grm = np.zeros((self.N, self.n_ue))
    self.m_bss2bss_adj = np.zeros((self.N, self.N))

    # sim intrinsics
    self.a_ues_power_index = np.zeros((self.n_ue), dtype=np.int32)
    self.a_ues_power = np.zeros((self.n_ue))
    self.m_ues2bss_gain = np.zeros((self.n_ue, self.N))
    self.m_ues2bss_path_loss = np.zeros((self.n_ue, self.N))
    self.a_ues_noise = np.repeat(self.sigma2, repeats=(self.n_ue))

    # others
    self.a_ues_indices = np.arange(self.n_ue, dtype=np.int32)
    self.a_bss_indices = np.arange(self.N, dtype=np.int32)
    self.m_ues2bss_h = np.zeros((self.n_ue, self.N))
    self.m_bss2iid_cands = np.zeros((self.N, self.n_ii))

    # counters, 0:step 1:total
    self.a_bss_request_counter = np.zeros((2, self.N), dtype=np.int64)
    self.a_ues_request_counter = np.zeros((2, self.n_ue), dtype=np.int64)
    self.a_bss_cache_hit_counter = np.zeros((2, self.N), dtype=np.int64)
    self.a_bss_cache_insert_counter = np.zeros((2, self.N), dtype=np.int64)
    self.a_bss_cache_popped_counter = np.zeros((2, self.N), dtype=np.int64)

    # begin setup
    print('setting up environment')

    print('configuring bs coordinates:', config.env.bs_topo)
    if False:
      pass
    elif config.env.bs_topo == 'grid':
      topo_gen = self.topo_factory_grid
      ymul = 1
    elif config.env.bs_topo == 'hex':
      topo_gen = self.topo_factory_hexlattice
      ymul = SIN60
    else:
      raise ValueError('invalid topo enum: {}'.format(config.env.bs_topo))

    self.v_bss_pos[:, :] = topo_gen(self.bs_nx, self.bs_ny)
    self.v_bss_pos[:, 0] *= self.L * (self.bs_nx - 1) * 2
    self.v_bss_pos[:, 1] *= self.L * (self.bs_ny - 1) * 2

    print('configuring ue coordinates:', config.env.ues_distribution)
    if False:
      pass
    elif config.env.ues_distribution == 'uniform':
      self.v_ues_pos[:, :2] = np.random.uniform(-.5, .5, size=(self.n_ue, 2))
      self.v_ues_pos[:, :2] += np.random.uniform(-.05, .05, size=(self.n_ue, 2))
      self.v_ues_pos[:, 0] *= self.L * (self.bs_nx) * 2
      self.v_ues_pos[:, 1] *= self.L * (self.bs_ny) * 2 * ymul
    elif config.env.ues_distribution == 'centers':
      indices = self.a_ues_indices.copy()
      np.random.shuffle(indices)
      indices = np.array_split(indices, self.N)
      for i in range(self.N):
        ues_indices = indices[i]
        bs_pos = self.v_bss_pos[i]
        ue_pos_t = np.random.uniform(.0, 1.0, size=ues_indices.shape) * 2 * np.pi
        ue_pos_r = np.random.uniform(.0, 1.0, size=ues_indices.shape)
        ue_pos_r = np.sqrt(ue_pos_r) * self.L * (np.pi/3)
        ue_pos_x = ue_pos_r * np.cos(ue_pos_t)
        ue_pos_y = ue_pos_r * np.sin(ue_pos_t)
        self.v_ues_pos[ues_indices] *= 0
        self.v_ues_pos[ues_indices] += bs_pos
        self.v_ues_pos[ues_indices, 0] += ue_pos_x
        self.v_ues_pos[ues_indices, 1] += ue_pos_y
    else:
      raise ValueError('invalid ue distribution enum: {}'.format(config.env.ues_distribution))

    self.m_ues2bss_h += np.random.rayleigh(self.rayleigh_scale, size=(self.n_ue, self.N))

    print('instantiating cache instances')
    if False:
      pass
    elif config.cache.type == 'fifo':
      cache_factory = FIFOCache
    elif config.cache.type == 'lifo':
      cache_factory = LIFOCache
    else:
      raise ValueError('invalid cache factory enum: {}'.format(config.cache.type))

    self.l_bss_caches = list()
    for bsi in range(self.N):
      cache = cache_factory(self.timer, config.cache.maxsize, config.cache.maxage)
      cache.insert_counter = self.a_bss_cache_insert_counter[:, bsi]
      cache.popped_counter = self.a_bss_cache_popped_counter[:, bsi]
      self.l_bss_caches.append(cache)

    print('calculating station assignments')
    self.update()

    print('saving initial states')
    self._attrs_t0_cp = {
      'v_ues_pos': self.v_ues_pos.copy(),
      'v_ues_hea': self.v_ues_hea.copy(),
      'v_ues_vel': self.v_ues_vel.copy(),
      'v_bss_pos': self.v_bss_pos.copy(),
      'v_bss_hea': self.v_bss_hea.copy(),
      'v_bss_vel': self.v_bss_vel.copy(),
      'a_ues_power_index': self.a_ues_power_index.copy(),
      'm_bss2iid_cands': self.m_bss2iid_cands.copy(),
    }

    print('done')
    print()

  def query_interfering_ues(self, ts):
    dfv = self.request_w.df[['ts', 'uid']]
    return dfv[dfv['ts'] == ts]['uid'].drop_duplicates().to_numpy()


  def update_dist(self):
    if self._do_skip_update():
      return
    self._do_update_dist = False
    self.m_bss2bss_dpos *= 0
    self.m_ues2ues_dpos *= 0
    self.m_ues2bss_dpos *= 0
    self.m_bss2bss_dpos += self.v_bss_pos
    self.m_ues2ues_dpos += self.v_ues_pos
    self.m_ues2bss_dpos += self.v_bss_pos
    self.m_bss2bss_dpos -= np.expand_dims(self.v_bss_pos, axis=1)
    self.m_ues2ues_dpos -= np.expand_dims(self.v_ues_pos, axis=1)
    self.m_ues2bss_dpos -= np.expand_dims(self.v_ues_pos, axis=1)
    np.sqrt(np.sum(np.square(self.m_bss2bss_dpos), axis=2), out=self.m_bss2bss_dist)
    np.sqrt(np.sum(np.square(self.m_ues2ues_dpos), axis=2), out=self.m_ues2ues_dist)
    np.sqrt(np.sum(np.square(self.m_ues2bss_dpos), axis=2), out=self.m_ues2bss_dist)
    self.m_ues2bss_dist.argmin(axis=1, out=self.a_ues_gr)
    np.equal(np.repeat(np.expand_dims(self.a_ues_gr, axis=0), self.N, axis=0), np.expand_dims(self.a_bss_indices, axis=1), out=self.m_bss2ues_grm)
    self.m_bss2bss_adj = self.m_bss2bss_dist <= (self.L * 2 + ATOL)

  def update_pos(self):
    if self._do_skip_update():
      return
    dt = self.timer.delta_time()
    self._do_update_dist |= np.isclose(dt, 0)
    self.v_bss_pos += self.v_bss_vel * dt
    self.v_ues_pos += self.v_ues_vel * dt

  def update_sim_intrinsics(self):
    self.m_ues2bss_path_loss = self.a * 10 * np.log10(self.m_ues2bss_dist)
    self.a_ues_power = self.a_power_set[self.a_ues_power_index]
    self.m_ues2bss_gain = self.m_ues2bss_path_loss / self.m_ues2bss_h

  def update(self):
    self.update_pos()
    self.update_dist()
    self.update_sim_intrinsics()

  def _do_skip_update(self):
    check = self.counter > 0
    check &= self.f_static
    check &= not self._do_update_dist
    return check

  def step(self): 
    # clear per-step counters
    self.a_ues_request_counter[0, :] *= 0
    self.a_bss_request_counter[0, :] *= 0
    self.a_bss_cache_hit_counter[0, :] *= 0
    for cache in self.l_bss_caches:
      cache.reset_counter()
    # prepare request batch
    batch = self.request_w.get_next_batch()
    for uid, iid, val, ts in batch.itertuples(index=False):
      # update sim
      self.timer.update(ts)
      self.update()
      # query bs index
      bsi = self.a_ues_gr[uid]
      # increment request counter on each bs and ue
      self.a_ues_request_counter[:, uid] += 1
      self.a_bss_request_counter[:, bsi] += 1
      # query cache instance
      cache = self.l_bss_caches[bsi]
      # record cache hit
      if cache.has(iid):
        self.a_bss_cache_hit_counter[:, bsi] += 1
      # tbd: recsys goes here
      cache.add(iid, iid, self.a_content_size[iid], None)
    # run eviction policy routine
    for cache in self.l_bss_caches:
      cache.evict()

  def reset(self):
    self.request_w.reset()
    for cache in self.l_bss_caches:
      cache.reset()
    for attr, cp in self._attrs_t0_cp.items():
      i_attr = getattr(self, attr).view()
      i_attr *= 0
      i_attr += cp
    self.update()
    self.a_bss_request_counter *= 0
    self.a_ues_request_counter *= 0

  @property
  def counter(self):
    return self.request_w.counter

  @property
  def stop_flag(self):
    return self.request_w.stop_flag

  @property
  def a_ues_power_lin(self):
    return 10**(self.a_ues_power/10)

  def plot_ax(self, ax: plt.Axes):
    for bs_pos in self.v_bss_pos[:, :2]:
      patch = plt.Circle(bs_pos, self.L, facecolor='#00000000', edgecolor='#232323', clip_on=False, linestyle='--')
      ax.add_patch(patch)
    for i_bs in range(self.N):
      ues_pos = self.v_ues_pos[self.a_ues_gr==i_bs]
      ues_bss_dist = self.m_ues2bss_dist[:, i_bs][self.a_ues_gr==i_bs]
      ues_bss_dist = np.clip(ues_bss_dist, 0.05, ues_bss_dist.max())
      ues_bss_dist = 1 / ues_bss_dist * (ues_bss_dist.max() - ues_bss_dist.min()) * 10
      ax.scatter(ues_pos[:, 0], ues_pos[:, 1], marker='x', s=ues_bss_dist)
    ax.scatter(self.v_bss_pos[:, 0], self.v_bss_pos[:, 1], marker='v', c='#ff0000')
    xliml, xlimu = ax.get_xlim()
    yliml, ylimu = ax.get_ylim()
    ax.set_xlim(min(xliml, yliml), max(xlimu, ylimu))
    ax.set_ylim(min(xliml, yliml), max(xlimu, ylimu))

