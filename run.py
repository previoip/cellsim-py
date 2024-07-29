import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellsim.config import Config, fmt
from cellsim.environ import CellNetEnviron
import torch
import traceback
import tqdm
import time

def init_seed(seed, reproducibility=True):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if reproducibility:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

class Main:
  def __init__(self):
    config_filepath = './config.ini'
    if len(sys.argv) > 1:
      path = sys.argv[1]
      if os.path.exists(path) and os.path.isfile(path):
        config_filepath = path
    print('using', config_filepath)
    self.config = Config(config_filepath)
    self.config.load()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def onstart(self):
    os.makedirs(self.config.default.results_subdir, exist_ok=True)
    init_seed(self.config.default.seed)
    self.df_request = pd.read_csv(os.path.join(self.config.dataset_settings.data_dir, self.config.dataset_settings.data_csv_requests), sep=';', low_memory=True)
    self.env = CellNetEnviron(self.df_request, self.config)

  def main(self):
    self.env.reset()
    tqdm_h = tqdm.tqdm(total=self.env.n_steps)
    while not self.env.stop_flag:
      self.env.step()
      tqdm_h.update(1)

  def onexit(self):
    self.config.save()
    fig, ax = plt.subplots(tight_layout=True)
    self.env.plot_ax(ax)
    fig.savefig('{}/{}_env.png'.format(self.config.default.results_subdir, fmt(self.config)))

if __name__ == '__main__':
  main = Main()
  main.onstart()
  try:
    main.main()
  except Exception:
    print(traceback.format_exc(), file=sys.stderr)
  except KeyboardInterrupt:
    pass
  main.onexit()