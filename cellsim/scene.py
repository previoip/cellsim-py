import numpy as np
import matplotlib.pyplot as plt
import enum

SIN60 = np.sin(np.deg2rad(60))
ATOL = 1e-8
LN2 = np.log(2)


class enum_bss_distribution(enum.StrEnum):
  hex = 'hex'
  grid = 'grid'

class enum_ues_distribution(enum.StrEnum):
  centers = 'centers'
  uniform = 'uniform'


class Scene:

  class rtfl(enum.Flag):
    default = 0
    skip_init = enum.auto()
    skip_dist_update = enum.auto()

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

  @staticmethod
  def zeroarr(a):
    a *= 0
    return a

  @staticmethod
  def log102lin(a, o=None):
    return np.power(10, a, out=o)

  @staticmethod
  def lin2log10(a, o=None):
    return np.log10(a, out=o)

  @classmethod
  def dBm2W(cls, a, o=None):
    return cls.log102lin((a-30)/10, o=o)

  @classmethod
  def dBm2mW(cls, a, o=None):
    return np.power(10, a/10)

  @staticmethod
  def W2dBm(cls, a, o=None):
    r = 10 * np.log10(a)
    r += 30
    if not o is None: np.copyto(o, r)
    return r

  @classmethod
  def mW2dBm(cls, a, o=None):
    r = 10 * np.log10(a)
    if not o is None: np.copyto(o, r)
    return r


  def __init__(
      self,
      bs_nx: int,
      bs_ny: int,
      n_uid: int,
      n_rb: int,
      n_ch: int,
      L: float,
      min_P: float,
      max_P: float,
      min_dist: float = 3,
      max_dist: float = 500,
      a: float = -3.5,
      flags: int = rtfl.default
   ):

    print('using bs_nx        :', bs_nx)
    print('using bs_ny        :', bs_ny)
    print('using n_uid        :', n_uid)
    print('using n_rb         :', n_rb)
    print('using n_ch         :', n_ch)
    print('using L (m)        :', L)
    print('using min_P (dBm)  :', min_P, '\t(mW):', self.dBm2mW(min_P))
    print('using max_P (dBm)  :', max_P, '\t(mW):', self.dBm2mW(max_P))
    print('using min_dist (m) :', min_dist)
    print('using max_dist (m) :', max_dist)
    print('using a            :', a)
    print('runtime flags      :', flags)
    print()

    # instances
    self.timer = TimestepTimer()

    # scalars
    self.s_min_dist_m = min_dist
    self.s_max_dist_m = max_dist
    self.s_n_uid_i = n_uid
    self.s_nx_bs_i = bs_nx
    self.s_ny_bs_i = bs_ny
    self.s_N_bs_i = self.s_nx_bs_i * self.s_ny_bs_i
    self.s_L_m = L
    self.s_atten_f = a
    self.c_n_rb_i = n_rb
    self.c_n_ch_i = n_ch
    self.s_min_power_dBm = min_P
    self.s_max_power_dBm = max_P
    self.s_min_power_mW = self.dBm2mW(min_P)
    self.s_max_power_mW = self.dBm2mW(max_P)
    self.s_rtfl = flags
    self.s_ymul_f = 1
    self.s_rt_updatecounter = 0
    # constants
    self.c_rayleigh_scale_f = 1

    # vectors
    #   *_pos: cart coord [x, y, z]
    #   *_hea: heading (antenna) [elev, azim]
    #   *_vel: cart velocity [u, v, w]
    print('allocating vectors')
    self.v_ues_pos = np.zeros((self.s_n_uid_i, 3))
    self.v_ues_hea = np.zeros((self.s_n_uid_i, 2))
    self.v_ues_vel = np.zeros((self.s_n_uid_i, 3))
    self.v_bss_pos = np.zeros((self.s_N_bs_i, 3))
    self.v_bss_hea = np.zeros((self.s_N_bs_i, 2))
    self.v_bss_vel = np.zeros((self.s_N_bs_i, 3))

    # pre-alloc arrays/matrix
    #   *_d*: delta *
    #   *_dist : sqrt(square(*_dl))
    #   *_gr: group indices
    #   *_grm: group mask
    #   *_adjm: adjacent mask
    print('allocating arrays/matrixes')
    self.m_bss2bss_dpos_m = np.zeros((self.s_N_bs_i, self.s_N_bs_i, 3))
    self.m_ues2ues_dpos_m = np.zeros((self.s_n_uid_i, self.s_n_uid_i, 3))
    self.m_ues2bss_dpos_m = np.zeros((self.s_n_uid_i, self.s_N_bs_i, 3))
    self.m_bss2bss_dist_m = np.zeros((self.s_N_bs_i, self.s_N_bs_i))
    self.m_ues2ues_dist_m = np.zeros((self.s_n_uid_i, self.s_n_uid_i))
    self.m_ues2bss_dist_m = np.zeros((self.s_n_uid_i, self.s_N_bs_i))

    # index
    self.a_ues_indices_i = np.arange(self.s_n_uid_i, dtype=np.int64)
    self.a_bss_indices_i = np.arange(self.s_N_bs_i, dtype=np.int64)

    # group mask
    self.a_ues_bs_gr_i = np.zeros((self.s_n_uid_i), dtype=np.int64)
    self.a_ues_rb_gr_i = np.zeros((self.s_n_uid_i), dtype=np.int64)
    self.a_ues_ch_gr_i = np.zeros((self.s_n_uid_i), dtype=np.int64)
    self.m_bss2bss_adjm_b = np.zeros((self.s_N_bs_i, self.s_N_bs_i), dtype=bool)
    self.m_ues2bss_grm_b = np.zeros((self.s_n_uid_i, self.s_N_bs_i), dtype=bool)

    # intrinsics
    self.m_ues2ues_fading_dBm = np.zeros((self.s_n_uid_i, self.s_n_uid_i))
    self.m_ues2bss_fading_dBm = np.zeros((self.s_n_uid_i, self.s_N_bs_i))
    self.m_ues2ues_fading_mW  = np.zeros((self.s_n_uid_i, self.s_n_uid_i))
    self.m_ues2bss_fading_mW  = np.zeros((self.s_n_uid_i, self.s_N_bs_i))
    self.a_ues_band_Hz        = np.zeros((self.s_n_uid_i))
    self.a_ues_power_dBm      = np.zeros((self.s_n_uid_i))
    self.a_ues_noise_dBm      = np.zeros((self.s_n_uid_i))
    self.a_ues_power_mW       = np.zeros((self.s_n_uid_i))
    self.a_ues_noise_mW       = np.zeros((self.s_n_uid_i))
    self.a_ues_intpf_f        = np.zeros((self.s_n_uid_i))
    self.m_ues2bss_h_f        = np.random.rayleigh(self.c_rayleigh_scale_f, size=(self.s_n_uid_i, self.s_N_bs_i))
    self.m_ues2bss_g_f        = np.zeros((self.s_n_uid_i, self.s_N_bs_i))
    self.m_ues2bss_g_ln       = np.zeros((self.s_n_uid_i, self.s_N_bs_i))
    self.m_ues2ues_h_f        = np.random.rayleigh(self.c_rayleigh_scale_f, size=(self.s_n_uid_i, self.s_n_uid_i))
    self.m_ues2ues_g_f        = np.zeros((self.s_n_uid_i, self.s_n_uid_i))
    self.m_ues2ues_g_ln       = np.zeros((self.s_n_uid_i, self.s_n_uid_i))
    self.a_ues_A_snr_f        = np.zeros((self.s_n_uid_i))
    self.a_ues_A_sii_f        = np.zeros((self.s_n_uid_i))
    self.a_ues_A_sinr_f       = np.zeros((self.s_n_uid_i))
    self.a_ues_snr_f          = np.zeros((self.s_n_uid_i))
    self.a_ues_sii_f          = np.zeros((self.s_n_uid_i))
    self.a_ues_sinr_f         = np.zeros((self.s_n_uid_i))
    self.a_ues_snr_dB         = np.zeros((self.s_n_uid_i))
    self.a_ues_sii_dB         = np.zeros((self.s_n_uid_i))
    self.a_ues_sinr_dB        = np.zeros((self.s_n_uid_i))
    self.a_ues_B_snr_f        = np.zeros((self.s_n_uid_i))
    self.a_ues_B_sii_f        = np.zeros((self.s_n_uid_i))
    self.a_ues_B_sinr_f       = np.zeros((self.s_n_uid_i))
    self.a_ues_Dd_snr_f       = np.zeros((self.s_n_uid_i))
    self.a_ues_Dd_sii_f       = np.zeros((self.s_n_uid_i))
    self.a_ues_Dd_sinr_f      = np.zeros((self.s_n_uid_i))


  def init_bss_distribution(self, topo):
    print('configuring bs coordinates:', topo)
    if False: pass
    elif topo == 'grid':
      topo_gen = self.topo_factory_grid
      self.s_ymul_f = 1
    elif topo == 'hex':
      topo_gen = self.topo_factory_hexlattice
      self.s_ymul_f = SIN60
    else:
      raise ValueError('invalid topo enum: {}'.format(topo))
    self.v_bss_pos[:, :] += topo_gen(self.s_nx_bs_i, self.s_ny_bs_i)
    self.v_bss_pos[:, 0] *= self.s_L_m * (self.s_nx_bs_i - 1) * 2
    self.v_bss_pos[:, 1] *= self.s_L_m * (self.s_ny_bs_i - 1) * 2
    self.update_spatial()

  def init_ues_distribution(self, distrib_type):
    print('configuring ue coordinates:', distrib_type)
    if False: pass
    elif distrib_type == 'uniform':
      self.v_ues_pos[:, :2] = np.random.uniform(-.5, .5, size=(self.s_n_uid_i, 2))
      self.v_ues_pos[:, 0] *= self.s_L_m * (self.s_nx_bs_i) * 2
      self.v_ues_pos[:, 1] *= self.s_L_m * (self.s_ny_bs_i) * 2 * self.s_ymul_f
    elif distrib_type == 'centers':
      indices = self.a_ues_indices_i.copy()
      np.random.shuffle(indices)
      indices = np.array_split(indices, self.s_N_bs_i)
      for i in range(self.s_N_bs_i):
        ues_indices_i = indices[i]
        bs_pos = self.v_bss_pos[i]
        ue_pos_t = np.random.uniform(.0, .9, size=ues_indices_i.shape) * 2 * np.pi
        ue_pos_r = np.random.uniform(.0, .9, size=ues_indices_i.shape)
        ue_pos_r = np.sqrt(ue_pos_r) * self.s_L_m * (np.pi/3)
        ue_pos_x = ue_pos_r * np.cos(ue_pos_t)
        ue_pos_y = ue_pos_r * np.sin(ue_pos_t)
        self.v_ues_pos[ues_indices_i] *= 0
        self.v_ues_pos[ues_indices_i] += bs_pos
        self.v_ues_pos[ues_indices_i, 0] += ue_pos_x
        self.v_ues_pos[ues_indices_i, 1] += ue_pos_y
    else:
      raise ValueError('invalid ue distribution enum: {}'.format(distrib_type))
    self.update_spatial()

  def set_ues_noise_dBm(self, sigma2):
    self.a_ues_noise_dBm[:] = sigma2
    self.a_ues_noise_mW[:] = self.dBm2mW(sigma2)

  def set_ues_bandwidth(self, b):
    self.a_ues_band_Hz[:] = b

  def update_spatial(self):
    dt = self.timer.delta_time()
    self.v_bss_pos += self.v_bss_vel * dt
    self.v_ues_pos += self.v_ues_vel * dt
    self.zeroarr(self.m_bss2bss_dpos_m)
    self.zeroarr(self.m_ues2ues_dpos_m)
    self.zeroarr(self.m_ues2bss_dpos_m)
    self.m_bss2bss_dpos_m += self.v_bss_pos
    self.m_ues2ues_dpos_m += self.v_ues_pos
    self.m_ues2bss_dpos_m += self.v_bss_pos
    self.m_bss2bss_dpos_m -= np.expand_dims(self.v_bss_pos, axis=1)
    self.m_ues2ues_dpos_m -= np.expand_dims(self.v_ues_pos, axis=1)
    self.m_ues2bss_dpos_m -= np.expand_dims(self.v_ues_pos, axis=1)
    self.m_bss2bss_dist_m[:, :] = np.linalg.norm(self.m_bss2bss_dpos_m, ord=2, axis=2)
    self.m_ues2ues_dist_m[:, :] = np.linalg.norm(self.m_ues2ues_dpos_m, ord=2, axis=2)
    self.m_ues2bss_dist_m[:, :] = np.linalg.norm(self.m_ues2bss_dpos_m, ord=2, axis=2)
    np.clip(self.m_bss2bss_dist_m, 1, 100e3, out=self.m_bss2bss_dist_m)
    np.clip(self.m_ues2ues_dist_m, self.s_min_dist_m, self.s_max_dist_m, out=self.m_ues2ues_dist_m)
    np.clip(self.m_ues2bss_dist_m, self.s_min_dist_m, self.s_max_dist_m, out=self.m_ues2bss_dist_m)
    # components dependent to spatial changes
    self.m_ues2bss_dist_m.argmin(axis=1, out=self.a_ues_bs_gr_i)
    self.m_bss2bss_adjm_b[:, :] = np.isclose(self.m_bss2bss_dist_m, self.s_L_m * 2, 1e-2)
    # update large scale fading component
    self.m_ues2ues_fading_mW[:, :] = np.power(self.m_ues2ues_dist_m/1000, self.s_atten_f)
    self.m_ues2bss_fading_mW[:, :] = np.power(self.m_ues2bss_dist_m/1000, self.s_atten_f)
    self.mW2dBm(self.m_ues2ues_fading_mW, o=self.m_ues2ues_fading_dBm)
    self.mW2dBm(self.m_ues2bss_fading_mW, o=self.m_ues2bss_fading_dBm)

  def update_intrinsics(self):
    for rbi in range(self.c_n_rb_i):
      rbm = self.a_ues_rb_gr_i == rbi
      ues = self.a_ues_indices_i[rbm]
      self.a_ues_ch_gr_i[ues] = np.random.uniform()
      ptot = self.a_ues_power_mW[rbm].sum()
      if ptot > self.s_max_power_mW:
        f = self.s_max_power_mW / ptot
        self.a_ues_power_mW[ues] *= f
        self.a_ues_power_dBm[ues] = self.mW2dBm(self.a_ues_power_mW[ues])
    np.clip(self.a_ues_power_mW, self.s_min_power_mW, self.s_max_power_mW, out=self.a_ues_power_mW)
    self.mW2dBm(self.a_ues_power_mW, o=self.a_ues_power_dBm)
    self.mW2dBm(self.a_ues_noise_mW, o=self.a_ues_noise_dBm)
    self.m_ues2bss_g_f[:, :] = np.power(self.m_ues2bss_h_f, 2) * self.m_ues2bss_fading_mW
    self.m_ues2ues_g_f[:, :] = np.power(self.m_ues2ues_h_f, 2) * self.m_ues2ues_fading_mW
    self.lin2log10(self.m_ues2bss_g_f, o=self.m_ues2bss_g_ln)
    self.lin2log10(self.m_ues2ues_g_f, o=self.m_ues2ues_g_ln)
    for uid in self.a_ues_indices_i:
      self.sub_update_ue_signal_metric(uid)

  def sub_update_ue_intrinsic(self, uid):
    ue_ii = self.get_ue_iinfo(uid)
    ptot = self.a_ues_power_mW[ue_ii.o_uid_rb].sum()
    if ptot > self.s_max_power_mW:
      f = self.s_max_power_mW / ptot
      self.a_ues_power_mW[ue_ii.o_uid_rb] *= f
      self.a_ues_power_dBm[ue_ii.o_uid_rb] = self.mW2dBm(self.a_ues_power_mW[ue_ii.o_uid_rb])
    self.a_ues_power_dBm[uid] = self.mW2dBm(self.a_ues_power_mW[uid])
    self.a_ues_noise_dBm[uid] = self.mW2dBm(self.a_ues_noise_mW[uid])
    self.m_ues2bss_g_f[uid, ue_ii.bsi] = np.power(self.m_ues2bss_h_f[uid, ue_ii.bsi], 2) * self.m_ues2bss_fading_mW[uid, ue_ii.bsi]
    self.m_ues2ues_g_f[uid, ue_ii.o_uid_rb] = np.power(self.m_ues2ues_h_f[uid, ue_ii.o_uid_rb], 2) * self.m_ues2ues_fading_mW[uid, ue_ii.o_uid_rb]
    self.m_ues2bss_g_ln[uid, ue_ii.bsi] = self.lin2log10(self.m_ues2bss_g_f[uid, ue_ii.bsi])
    self.m_ues2ues_g_ln[uid, ue_ii.o_uid_rb] = self.lin2log10(self.m_ues2ues_g_f[uid, ue_ii.o_uid_rb])

  def sub_update_ue_signal_metric(self, uid):
    ue_ii = self.get_ue_iinfo(uid)
    ues_intrf = ue_ii.o_uid_rbch
    interf_p = self.a_ues_power_mW[ues_intrf]
    interf_g = self.m_ues2ues_g_f[uid, ues_intrf]
    interf_pf = interf_p * interf_g
    self.a_ues_intpf_f[uid] = interf_pf.sum()
    # snr sii sinr
    g = self.m_ues2bss_g_f[uid, ue_ii.bsi]
    p = self.a_ues_power_mW[uid]
    n = self.a_ues_noise_mW[uid]
    i = self.a_ues_intpf_f[uid]
    self.a_ues_A_snr_f[uid] = g / (1+n)
    self.a_ues_A_sii_f[uid] = g / (1+i)
    self.a_ues_A_sinr_f[uid] = g / (1+n+i)
    self.a_ues_snr_f[uid] = self.a_ues_A_snr_f[uid] * p
    self.a_ues_sii_f[uid] = self.a_ues_A_sii_f[uid] * p
    self.a_ues_sinr_f[uid] = self.a_ues_A_sinr_f[uid] * p
    self.a_ues_snr_dB[uid] = self.lin2log10(self.a_ues_snr_f[uid])
    self.a_ues_sii_dB[uid] = self.lin2log10(self.a_ues_sii_f[uid])
    self.a_ues_sinr_dB[uid] = self.lin2log10(self.a_ues_sinr_f[uid])
    self.a_ues_B_snr_f[uid] = self.a_ues_band_Hz[uid] * np.log2(1 + self.a_ues_snr_f[uid])
    self.a_ues_B_sii_f[uid] = self.a_ues_band_Hz[uid] * np.log2(1 + self.a_ues_sii_f[uid])
    self.a_ues_B_sinr_f[uid] = self.a_ues_band_Hz[uid] * np.log2(1 + self.a_ues_sinr_f[uid])
    return ue_ii

  def update_rb(self):
    for bsi in self.a_bss_indices_i:
      bs_ii = self.get_bs_iinfo(bsi)
      ues = bs_ii.ues.copy()
      np.random.shuffle(ues)
      for rbi, ues_rb in enumerate(np.array_split(ues, self.c_n_rb_i)):
        self.a_ues_rb_gr_i[ues_rb] = rbi
      self.a_ues_rb_gr_i[ues] = np.random.randint(0, self.c_n_ch_i, size=(len(ues)))

  def update(self):
    if self.s_rt_updatecounter == 0 or (self.s_rtfl & self.rtfl.skip_dist_update):
      self.update_spatial()
      self.update_rb()
    self.update_intrinsics()
    self.s_rt_updatecounter += 1

  def get_bs_iinfo(self, bsi):
    t_bs_iinfo = namedtuple('BSIndicesInfo', ['bsi', 'ues', 'bsi_adjacent'])
    uem = self.a_ues_bs_gr_i == bsi
    uids = self.a_ues_indices_i[uem]
    bsam = self.m_bss2bss_adjm_b[bsi]
    bsi_adj = self.a_bss_indices_i[bsam]
    return  t_bs_iinfo(bsi, uids, bsi_adj)

  def get_ue_iinfo(self, uid):
    t_ue_iinfo = namedtuple('UEIndicesInfo', ['uid', 'bsi', 'o_uid_bs', 'o_uid_rb', 'o_uid_ch', 'o_uid_rbch'])
    bsi = self.a_ues_bs_gr_i[uid]
    rb = self.a_ues_rb_gr_i[uid]
    ch = self.a_ues_ch_gr_i[uid]
    uem = (self.a_ues_bs_gr_i == bsi) & (self.a_ues_indices_i != uid)
    rbm = self.a_ues_rb_gr_i == rb
    chm = self.a_ues_ch_gr_i == ch
    uid_bs = self.a_ues_indices_i[uem]
    uid_rb = self.a_ues_indices_i[uem & rbm]
    uid_ch = self.a_ues_indices_i[uem & chm]
    uid_rbch = self.a_ues_indices_i[uem & rbm & chm]
    return t_ue_iinfo(uid, bsi, uid_bs, uid_rb, uid_ch, uid_rbch)

  def get_ues2bss_indice_splice(self):
    uids = self.a_ues_indices_i
    bsis = self.a_ues_bs_gr_i[uids]
    return (uids, bsis)

  def set_ue_power_mW(self, uid, p_mW):
    self.a_ues_power_mW[uid] = p_mW
    self.a_ues_power_dBm[uid] = self.mW2dBm(p_mW)
    self.sub_update_ue_intrinsic(uid)
    self.sub_update_ue_signal_metric(uid)

  def set_ue_power_dBm(self, uid, p_dBm):
    self.set_ue_power_mW(uid, self.dBm2mW(p_dBm))

  def reset(self):
    self.zeroarr(self.a_ues_power_mW)
    self.update_intrinsics()


def cellnet_plot_sigstrength(cn: Scene, pset=[-20, -10, 0, 10, 20, 30], fname='sigstrength'):
  print('generating plot signl strength')
  vs = [
    ('sinr', cn.a_ues_sinr_dB),
    # ('sii', cn.a_ues_sii_dB),
    ('snr', cn.a_ues_snr_dB),
  ]
  p0 = cn.a_ues_power_mW.copy()
  fig, axs = plt.subplots(2, len(vs), sharex=False, sharey=True, figsize=(4*len(vs)+1, 12))
  kws = {'alpha':.3, 'marker':'o', 'edgecolors':'none'}

  axs[0, 0].set_ylabel('strength (dB)')
  axs[1, 0].set_ylabel('strength (dB)')
  for nx, p in enumerate(pset):
    print(f'evaluating p: {p} dBm')
    for i in cn.a_ues_indices_i:
      cn.set_ue_power_dBm(i, p)
    cn.update()
    for i, (l, v) in enumerate(vs):
      ues = cn.a_ues_indices_i
      bss = cn.a_ues_bs_gr_i
      d = cn.m_ues2bss_dist_m[ues, bss]
      axs[0, i].title.set_text(l)
      axs[0, i].scatter(ues, v, label=f'{p} dBm', **kws)
      axs[0, i].grid(True)
      axs[0, i].set_xlabel('uid')
      axs[1, i].scatter(d, v, label=f'{p} dBm', **kws)
      axs[1, i].grid(True)
      axs[1, i].set_xlabel('dist (m)')
  fig.legend(*axs[-1, -1].get_legend_handles_labels())
  fig.tight_layout()
  fig.savefig(fname + '.png')
  cn.a_ues_power_mW[:] = p0
  cn.update()

def cellnet_plot_topo(cn: Scene, fname='topo'):
  fig, ax = plt.subplots(figsize=(8,8))
  for bs_pos in cn.v_bss_pos[:, :2]:
    patch = plt.Circle(bs_pos, cn.s_L_m, facecolor='#00000000', edgecolor='#232323', clip_on=False, linestyle='--')
    ax.add_patch(patch)
  for bsi in range(cn.s_N_bs_i):
    ues = cn.a_ues_indices_i[cn.a_ues_bs_gr_i==bsi]
    ues_pos = cn.v_ues_pos[ues]
    if len(ues_pos) == 0:
      continue
    ues_bss_dist = cn.m_ues2bss_dist_m[:, bsi][cn.a_ues_bs_gr_i==bsi]
    ues_bss_dist = np.clip(ues_bss_dist, 0.05, max(0.01, ues_bss_dist.max()))
    ues_bss_dist = 1 / ues_bss_dist * (ues_bss_dist.max() - ues_bss_dist.min()) * 10
    ax.scatter(ues_pos[:, 0], ues_pos[:, 1], marker='x', s=ues_bss_dist)
  ax.scatter(cn.v_bss_pos[:, 0], cn.v_bss_pos[:, 1], marker='v', c='#ff0000')
  xliml, xlimu = ax.get_xlim()
  yliml, ylimu = ax.get_ylim()
  ax.set_xlim(min(xliml, yliml), max(xlimu, ylimu))
  ax.set_ylim(min(xliml, yliml), max(xlimu, ylimu))
  fig.savefig(fname + '.png')
