import sys
import os
import urllib.parse
import pandas as pd
from zipfile import ZipFile
from tarfile import TarFile
from cellsim.config import Config
from cellsim.util.rutil.dl import downloader, new_session


if __name__ == '__main__':
  config_filepath = './config.ini'

  if len(sys.argv) > 1:
    path = sys.argv[1]
    if os.path.exists(path) and os.path.isfile(path):
      config_filepath = path

  print('using ', config_filepath)
  config = Config(config_filepath)
  config.load()

  dataset_url = urllib.parse.urljoin(
    config.dataset_info.host,
    '{}/{}'.format(
      config.dataset_info.path,
      config.dataset.archive_name
    )
  )

  download_folder = os.path.join(
    config.dataset_settings.download_dir,
    config.dataset_info.path
  )

  extract_folder = os.path.join(
    config.dataset_settings.extract_dir,
    config.dataset_info.path
  )

  archive_filepath = os.path.join(
    download_folder,
    config.dataset.archive_name
  )

  extract_filepath = os.path.join(
    extract_folder,
    config.default.dataset,
    config.dataset.data_file
  )

  os.makedirs(download_folder, exist_ok=True)
  os.makedirs(extract_folder, exist_ok=True)

  if not os.path.exists(archive_filepath) or not os.path.isfile(archive_filepath):
    sess = new_session()
    downloader.to_folder(sess, dataset_url, download_folder)
    sess.close()

  if not os.path.exists(extract_filepath) or not os.path.isfile(extract_filepath):
    if archive_filepath.endswith('.zip'):
      handler = ZipFile
    elif archive_filepath.endswith('.tar.gz') or \
      archive_filepath.endswith('.tar'):
      handler = TarFile
    else:
      raise FileNotFoundError('invalid extension given on downloaded file:', archive_filepath)

    with open(archive_filepath, 'rb') as fp:
      archive_handler = handler(fp)
      for f in archive_handler.filelist:
        if f.filename.endswith(config.dataset.data_file):
          archive_handler.extract(f, extract_folder)
          break
      else:
        raise FileNotFoundError('cannot retrieve file from archive:', config.dataset.data_file)

  delimiter = '\t' if config.dataset.delimiter == '\\t' else config.dataset.delimiter
  colnames = [getattr(config.dataset, 'colname_'+i) for i in config.dataset.colsequence.split('|')]
  
  with open(extract_filepath) as fp:
    df = pd.read_csv(
      fp,
      sep=delimiter,
      names=colnames,
      header=None if config.dataset.use_colnames_as_headers=='true' else 0,
      low_memory=True
    )
