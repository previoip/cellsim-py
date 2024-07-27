import numpy as np
import matplotlib.pyplot as plt
from cellsim.core.timer import TimestepTimer
from cellsim.cache import FIFOCache, LIFOCache
from cellsim.config import Config

SIN60 = np.sin(np.deg2rad(60))

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

  def __init__(self, n_ue, config: Config):
    self.n_ue = n_ue
    self.bs_nx = config.env.bs_nx
    self.bs_ny = config.env.bs_ny
    self.N = self.bs_nx * self.bs_ny
    self.L = config.env.cell_radius
    self.a = config.env.a
    self.sigma2 = config.env.sigma2
    self.rayleigh_scale = 1
    self.power_set = np.linspace(config.env.power_min, config.env.power_max, config.env.power_level)
    self.timer = TimestepTimer()
    self.bss_h = np.random.rayleigh(self.rayleigh_scale, size=(self.N))

    # vectors
    #   *_pos: coord
    #   *_hea: heading
    #   *_vel: velocity
    self.v_ues_pos = np.zeros((self.n_ue, 3))
    self.v_ues_hea = np.zeros((self.n_ue, 2))
    self.v_ues_vel = np.zeros((self.n_ue, 3))
    self.v_bss_pos = np.zeros((self.N, 3))
    self.v_bss_hea = np.zeros((self.N, 2))
    self.v_bss_vel = np.zeros((self.N, 3))

    # prealloc arrays/matrix
    #   *_d*: delta *
    #   *_dist : sqrt(square(*_dl))
    #   *_gr: group
    #   *_grm: group mask
    self.m_bss2bss_dpos = np.zeros((self.N, self.N, 3))
    self.m_ues2ues_dpos = np.zeros((self.n_ue, self.n_ue, 3))
    self.m_ues2bss_dpos = np.zeros((self.n_ue, self.N, 3))
    self.m_bss2bss_dist = np.zeros((self.N, self.N))
    self.m_ues2ues_dist = np.zeros((self.n_ue, self.n_ue))
    self.m_ues2bss_dist = np.zeros((self.n_ue, self.N))
    self.a_ues_gr = np.zeros((self.n_ue), dtype=np.int64)
    self.m_bss2ues_grm = np.zeros((self.N, self.n_ue))

    # sim intrinsics
    self.a_ues_power_index = np.zeros((self.n_ue), dtype=np.int64)
    self.a_ues_power = np.zeros((self.n_ue))
    self.a_ues_gain = np.zeros((self.n_ue, self.N))
    self.a_ues2bss_path_loss = np.zeros((self.n_ue, self.N))

    self.ues_i = np.arange(self.n_ue)
    self.bss_i = np.arange(self.N)

    if False:
      pass
    elif config.env.bs_topo == 'grid':
      topo_gen = self.topo_factory_grid
      ymul = 1
    elif config.env.bs_topo == 'hex':
      topo_gen = self.topo_factory_hexlattice
      ymul = SIN60
    else:
      ValueError('invalid value:', config.env.bs_topo)

    self.v_bss_pos[:, :] = topo_gen(self.bs_nx, self.bs_ny)
    self.v_bss_pos[:, 0] *= self.L * (self.bs_nx - 1) * 2
    self.v_bss_pos[:, 1] *= self.L * (self.bs_ny - 1) * 2

    self.v_ues_pos[:, :2] = np.random.uniform(-.5, .5, size=(self.n_ue, 2))
    self.v_ues_pos[:, :2] += np.random.uniform(-.05, .05, size=(self.n_ue, 2))
    self.v_ues_pos[:, 0] *= self.L * (self.bs_nx) * 2
    self.v_ues_pos[:, 1] *= self.L * (self.bs_ny) * 2 * ymul

    self.update()

    if False:
      pass
    elif config.cache.type == 'fifo':
      cache_factory = FIFOCache
    elif config.cache.type == 'lifo':
      cache_factory = LIFOCache
    else:
      raise ValueError('invalid cache factory enum: {}'.format(config.cache.type))

    self.bss_caches = [cache_factory(self.timer, config.cache.maxsize, config.cache.maxage) for _ in range(self.N)]


  def update_dist(self):
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
    np.equal(np.repeat(np.expand_dims(self.a_ues_gr, axis=0), self.bss_i.shape[0], axis=0), np.expand_dims(self.bss_i, axis=1), out=self.m_bss2ues_grm)

  def update_pos(self):
    dt = self.timer.delta_time()
    self.v_bss_pos += self.v_bss_vel * dt
    self.v_ues_pos += self.v_ues_vel * dt

  def update_env_states(self):
    self.a_ues_power = self.power_set[self.a_ues_power_index]
    self.a_ues2bss_path_loss = self.a * 10 * np.log10(self.m_ues2bss_dist)
    # self.a_ues_gain

  def update(self):
    self.update_pos()
    self.update_dist()
    self.update_env_states()

  @property
  def ues_power_lin(self):
    return 10**(self.a_ues_power/10)

  # @property
  # def ues_gain(self):
  #   return self.bss_h[self.a_ues_gr] / self.a_ues_power

  def plot_ax(self, ax: plt.Axes):
    for bs_pos in self.v_bss_pos[:, :2]:
      patch = plt.Circle(bs_pos, self.L, facecolor='#00000000', edgecolor='#232323', clip_on=False, linestyle='--')
      ax.add_patch(patch)
    for i_bs in range(self.N):
      v_pos_a_ues_gr = self.v_ues_pos[self.a_ues_gr==i_bs]
      m_dist_ues2bss_gr = self.m_ues2bss_dist[:, i_bs][self.a_ues_gr==i_bs]
      m_dist_ues2bss_gr = np.clip(m_dist_ues2bss_gr, 0.05, m_dist_ues2bss_gr.max())
      m_dist_ues2bss_gr = 1 / m_dist_ues2bss_gr * (m_dist_ues2bss_gr.max() - m_dist_ues2bss_gr.min()) * 10
      ax.scatter(v_pos_a_ues_gr[:, 0], v_pos_a_ues_gr[:, 1], marker='x', s=m_dist_ues2bss_gr)
    ax.scatter(self.v_bss_pos[:, 0], self.v_bss_pos[:, 1], marker='v', c='#ff0000')
    xliml, xlimu = ax.get_xlim()
    yliml, ylimu = ax.get_ylim()
    ax.set_xlim(min(xliml, yliml), max(xlimu, ylimu))
    ax.set_ylim(min(xliml, yliml), max(xlimu, ylimu))

