import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellsim.config import Config, fmt
from cellsim.environ import CellNetEnviron, RequestWrapper
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


if __name__ == '__main__':
  config_filepath = './config.ini'

  if len(sys.argv) > 1:
    path = sys.argv[1]
    if os.path.exists(path) and os.path.isfile(path):
      config_filepath = path

  print('using ', config_filepath)
  config = Config(config_filepath)
  config.load()

  os.makedirs(config.default.results_dir, exist_ok=True)
  init_seed(config.default.seed)

  df_request = pd.read_csv(os.path.join(config.dataset_settings.data_dir, config.dataset_settings.data_csv_requests), sep=';', low_memory=True)
  request_w = RequestWrapper(df_request, config)
  env = CellNetEnviron(request_w, config)
  env.reset()

  fig, ax = plt.subplots()
  env.plot_ax(ax)
  fig.savefig('{}/{}_env.png'.format(config.default.results_dir, fmt(config)))

  config.save()
