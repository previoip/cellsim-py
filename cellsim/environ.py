import numpy as np
from cellsim.core.timer import TimestepTimer

class Environ:

  @staticmethod
  def topo_factory_hexlattice(nx, ny):
    x = np.linspace(0, 1*(nx-1), nx)
    y = np.linspace(0, 1*(ny-1), ny)
    ax, ay = np.meshgrid(x, y)
    ax[1::2] += 0.5
    ay *= np.sin(np.deg2rad(60))
    ax, ay = ax.flatten(), ay.flatten()
    return np.column_stack((ax, ay, np.zeros(ax.shape)))

  def __init__(self, ue_n, bs_nx, bs_ny, L):
    self.ue_n = ue_n
    self.bs_nx = bs_nx
    self.bs_ny = bs_ny
    self.N = bs_nx * bs_ny
    self.L = L
    self.ues_pos = np.zeros((self.ue_n, 3), dtype=np.float64)
    self.ues_hea = np.zeros((self.ue_n, 2), dtype=np.float64)
    self.ues_vel = np.zeros((self.ue_n, 1), dtype=np.float64)
    self.bss_pos = np.zeros((self.bs_nx, self.bs_ny, 3), dtype=np.float64)
    self.bss_hea = np.zeros((self.bs_nx, self.bs_ny, 2), dtype=np.float64)
    self.bss_vel = np.zeros((self.bs_nx, self.bs_ny, 1), dtype=np.float64)
    self.timer = TimestepTimer()
