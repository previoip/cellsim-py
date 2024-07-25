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

  def __init__(self, bs_topo, ue_n, bs_nx, bs_ny, L, cache_maxsize, cache_maxage):
    self.ue_n = ue_n
    self.bs_nx = bs_nx
    self.bs_ny = bs_ny
    self.N = bs_nx * bs_ny
    self.L = L
    self.timer = TimestepTimer()
    self.ues_pos = np.zeros((self.ue_n, 3), dtype=np.float32)
    self.ues_hea = np.zeros((self.ue_n, 2), dtype=np.float32)
    self.ues_vel = np.zeros((self.ue_n, 3), dtype=np.float32)
    self.bss_pos = np.zeros((self.N, 3), dtype=np.float32)
    self.bss_hea = np.zeros((self.N, 2), dtype=np.float32)
    self.bss_vel = np.zeros((self.N, 3), dtype=np.float32)
    self.bss_dist = np.zeros((self.N, self.N), dtype=np.float32)
    self.ues_dist = np.zeros((self.ue_n, self.ue_n), dtype=np.float32)
    self.ues_bss_dist = np.zeros((self.ue_n, self.N), dtype=np.float32)
    self.bss_dist_fb = np.zeros((self.N, self.N, 3), dtype=np.float32)
    self.ues_dist_fb = np.zeros((self.ue_n, self.ue_n, 3), dtype=np.float32)
    self.ues_bss_dist_fb = np.zeros((self.ue_n, self.N, 3), dtype=np.float32)
    self.bss_caches = [FIFOCache(self.timer, cache_maxsize, cache_maxage) for _ in range(self.N)]

    if bs_topo == 'grid':
      topo_gen = self.topo_factory_grid
      ymul = 1
    elif bs_topo == 'hex':
      ymul = SIN60
      topo_gen = self.topo_factory_hexlattice
    else:
      ValueError('invalid value:', bs_topo)

    self.bss_pos[:, :] = topo_gen(self.bs_nx, self.bs_ny)
    self.bss_pos[:, 0] *= self.L * (self.bs_nx - 1) * 2
    self.bss_pos[:, 1] *= self.L * (self.bs_ny - 1) * 2

    self.ues_pos[:, :2] = np.random.uniform(-.5, .5, size=(self.ue_n, 2))
    self.ues_pos[:, 0] *= self.L * (self.bs_nx) * 2
    self.ues_pos[:, 1] *= self.L * (self.bs_ny) * 2 * ymul

  def update_dist(self):
    self.bss_dist_fb *= 0
    self.bss_dist_fb += self.bss_pos
    self.bss_dist_fb -= np.expand_dims(self.bss_pos, axis=1)
    self.bss_dist_fb[:, :, :] = np.square(self.bss_dist_fb)
    self.bss_dist[:, :] = np.sqrt(np.sum(self.bss_dist_fb, axis=2))

    self.ues_dist_fb *= 0
    self.ues_dist_fb += self.ues_pos
    self.ues_dist_fb -= np.expand_dims(self.ues_pos, axis=1)
    self.ues_dist_fb[:, :, :] = np.square(self.ues_dist_fb)
    self.ues_dist[:, :] = np.sqrt(np.sum(self.ues_dist_fb, axis=2))

    self.ues_bss_dist_fb *= 0
    self.ues_bss_dist_fb += self.bss_pos
    self.ues_bss_dist_fb -= np.expand_dims(self.ues_pos, axis=1)
    self.ues_bss_dist_fb[:, :, :] = np.square(self.ues_bss_dist_fb)
    self.ues_bss_dist[:, :] = np.sqrt(np.sum(self.ues_bss_dist_fb, axis=2))

  def update_pos(self):
    self.bss_pos += self.bss_vel * self.timer.delta_time()
    self.ues_pos += self.ues_vel * self.timer.delta_time()

  def update(self):
    self.update_dist()
    self.update_pos()

  def plot_ax(self, ax: plt.Axes):
    ax.scatter(self.ues_pos[:, 0], self.ues_pos[:, 1], marker='.')
    ax.scatter(self.bss_pos[:, 0], self.bss_pos[:, 1], marker='x')
    for bs_pos in self.bss_pos[:, :2]:
      patch = plt.Circle(bs_pos, self.L, facecolor='#00000000', edgecolor='#232323', clip_on=False)
      ax.add_patch(patch)
    xliml, xlimu = ax.get_xlim()
    yliml, ylimu = ax.get_ylim()
    ax.set_xlim(min(xliml, yliml), max(xlimu, ylimu))
    ax.set_ylim(min(xliml, yliml), max(xlimu, ylimu))
