import numpy as np
import matplotlib.pyplot as plt
from cellsim.core.timer import TimestepTimer
from cellsim.cache import FIFOCache

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

  def __init__(self, bs_topo, n_ue, bs_nx, bs_ny, L, cache_maxsize, cache_maxage):
    self.n_ue = n_ue
    self.bs_nx = bs_nx
    self.bs_ny = bs_ny
    self.N = bs_nx * bs_ny
    self.L = L
    self.timer = TimestepTimer()
    self.ues_pos = np.zeros((self.n_ue, 3))
    self.ues_hea = np.zeros((self.n_ue, 2))
    self.ues_vel = np.zeros((self.n_ue, 3))
    self.bss_pos = np.zeros((self.N, 3))
    self.bss_hea = np.zeros((self.N, 2))
    self.bss_vel = np.zeros((self.N, 3))
    self.bss_bss_dl = np.zeros((self.N, self.N, 3))
    self.ues_ues_dl = np.zeros((self.n_ue, self.n_ue, 3))
    self.ues_bss_dl = np.zeros((self.n_ue, self.N, 3))
    self.bss_bss_l = np.zeros((self.N, self.N))
    self.ues_ues_l = np.zeros((self.n_ue, self.n_ue))
    self.ues_bss_l = np.zeros((self.n_ue, self.N))
    self.ues_gr = np.zeros((self.n_ue), dtype=np.int64)
    self.bss_caches = [FIFOCache(self.timer, cache_maxsize, cache_maxage) for _ in range(self.N)]

    if bs_topo == 'grid':
      topo_gen = self.topo_factory_grid
      ymul = 1
    elif bs_topo == 'hex':
      topo_gen = self.topo_factory_hexlattice
      ymul = SIN60
    else:
      ValueError('invalid value:', bs_topo)

    self.bss_pos[:, :] = topo_gen(self.bs_nx, self.bs_ny)
    self.bss_pos[:, 0] *= self.L * (self.bs_nx - 1) * 2
    self.bss_pos[:, 1] *= self.L * (self.bs_ny - 1) * 2

    self.ues_pos[:, :2] = np.random.uniform(-.5, .5, size=(self.n_ue, 2))
    self.ues_pos[:, :2] += np.random.uniform(-.05, .05, size=(self.n_ue, 2))
    self.ues_pos[:, 0] *= self.L * (self.bs_nx) * 2
    self.ues_pos[:, 1] *= self.L * (self.bs_ny) * 2 * ymul

    self.update_dist()
    ues_oob = self.ues_bss_l > self.L
    print(ues_oob)

  def update_dist(self):
    self.bss_bss_dl *= 0
    self.ues_ues_dl *= 0
    self.ues_bss_dl *= 0
    self.bss_bss_dl += self.bss_pos
    self.ues_ues_dl += self.ues_pos
    self.ues_bss_dl += self.bss_pos
    self.bss_bss_dl -= np.expand_dims(self.bss_pos, axis=1)
    self.ues_ues_dl -= np.expand_dims(self.ues_pos, axis=1)
    self.ues_bss_dl -= np.expand_dims(self.ues_pos, axis=1)
    self.bss_bss_l[:, :] = np.sqrt(np.sum(np.square(self.bss_bss_dl), axis=2))
    self.ues_ues_l[:, :] = np.sqrt(np.sum(np.square(self.ues_ues_dl), axis=2))
    self.ues_bss_l[:, :] = np.sqrt(np.sum(np.square(self.ues_bss_dl), axis=2))
    self.ues_gr[:] = self.ues_bss_l.argmin(axis=1)

  def update_pos(self):
    dt = self.timer.delta_time()
    self.bss_pos += self.bss_vel * dt
    self.ues_pos += self.ues_vel * dt

  def update(self):
    self.update_dist()
    self.update_pos()

  def plot_ax(self, ax: plt.Axes):
    for bs_pos in self.bss_pos[:, :2]:
      patch = plt.Circle(bs_pos, self.L, facecolor='#00000000', edgecolor='#232323', clip_on=False, linestyle='--')
      ax.add_patch(patch)
    for i_bs in range(self.N):
      ues_pos_gr = self.ues_pos[self.ues_gr==i_bs]
      ues_bss_l_gr = self.ues_bss_l[:, i_bs][self.ues_gr==i_bs]
      ues_bss_l_gr = np.clip(ues_bss_l_gr, 0.05, ues_bss_l_gr.max())
      ues_bss_l_gr = 1 / ues_bss_l_gr * (ues_bss_l_gr.max() - ues_bss_l_gr.min()) * 10
      ax.scatter(ues_pos_gr[:, 0], ues_pos_gr[:, 1], marker='x', s=ues_bss_l_gr)
    ax.scatter(self.bss_pos[:, 0], self.bss_pos[:, 1], marker='v', c='#ff0000')
    xliml, xlimu = ax.get_xlim()
    yliml, ylimu = ax.get_ylim()
    ax.set_xlim(min(xliml, yliml), max(xlimu, ylimu))
    ax.set_ylim(min(xliml, yliml), max(xlimu, ylimu))

