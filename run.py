import sys
import os
from cellsim.config import Config


if __name__ == '__main__':
  config_filepath = './config.ini'

  if len(sys.argv) > 1:
    path = sys.argv[1]
    if os.path.exists(path) and os.path.isfile(path):
      config_filepath = path

  print('using ', config_filepath)
  config = Config(config_filepath)
  config.load()
