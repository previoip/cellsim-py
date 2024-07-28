import sys
import os
import urllib.parse
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from zipfile import ZipFile
from tarfile import TarFile
from cellsim.config import Config, fmt
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

  print('using dataset ', config.default.dataset)

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
    print('downloading dataset archive:', config.dataset.archive_name)
    hash_h = hashlib.md5()
    sess = new_session()
    downloader.to_folder(sess, dataset_url, download_folder, callback=hash_h.update)
    sess.close()
    if not hash_h.hexdigest() == config.dataset.hashinfo[config.dataset.archive_name].hash:
      raise ValueError('download error: file hash invalid')


  if not os.path.exists(extract_filepath) or not os.path.isfile(extract_filepath):
    print('extracting file from archive:', config.dataset.data_file)
    if archive_filepath.endswith('.zip'):
      handler = ZipFile
    elif archive_filepath.endswith('.tar.gz') or \
      archive_filepath.endswith('.tar'):
      handler = TarFile
    else:
      raise FileNotFoundError('invalid extension given on downloaded file: {}'.format(archive_filepath))

    with open(archive_filepath, 'rb') as fp:
      archive_handler = handler(fp)
      for f in archive_handler.filelist:
        if f.filename.endswith(config.dataset.data_file):
          archive_handler.extract(f, extract_folder)
          break
      else:
        raise FileNotFoundError('cannot retrieve file from archive: {}'.format(config.dataset.data_file))

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

  df = df.rename(columns={
    config.dataset.colname_uid: 'uid',
    config.dataset.colname_iid: 'iid',
    config.dataset.colname_inter: 'val',
    config.dataset.colname_ts: 'ts',
  })

  for colname in ['uid', 'iid']:
    df_t = df[[colname]].drop_duplicates().sort_values(by=colname)
    df_t['temp'] = df_t[colname].argsort()
    df = df.merge(df_t, how='left').drop([colname], axis=1).rename(columns={'temp':colname})

  df = df.sort_values(by='ts').reset_index(drop=True)
  df['ts'] -= df['ts'].min()
  df = df[['uid','iid','val','ts']]

  os.makedirs(config.dataset_settings.data_dir, exist_ok=True)
  data_filepath = os.path.join(config.dataset_settings.data_dir, config.dataset_settings.data_csv_requests)
  df.to_csv(data_filepath, encoding='utf8', sep=';', index=False)
  print('saved processed file:', data_filepath)

  df_uid = df[['uid']].drop_duplicates().sort_values(by='uid')
  df_uid['inter'] = df_uid['uid'].apply(lambda i: df[df['uid'] == i]['iid'].to_list())
  data_filepath = os.path.join(config.dataset_settings.data_dir, config.dataset_settings.data_csv_inter)
  df_uid.to_csv(data_filepath, encoding='utf8', sep=';', index=False)
  print('saved processed file:', data_filepath)


  print('generating dataset profile:', config.default.dataset)
  os.makedirs(config.default.results_subdir, exist_ok=True)
  n_bins = len(df) // config.default.request_per_step // 2

  fig, ax = plt.subplots(tight_layout=True)
  ax.hist(df['ts'], bins=n_bins)
  ax.set_title('request freq')
  ax.set_xlabel('time')
  ax.set_ylabel('iid')
  fig.savefig('{}/ts_freq.png'.format(config.default.results_subdir))

  fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
  ax1.hist2d(df['ts'], df['iid'], bins=n_bins)
  ax1.set_title('iid freq')
  ax1.set_xlabel('time')
  ax1.set_ylabel('iid')

  ax2.hist2d(df['uid'], df['iid'], bins=n_bins)
  ax2.set_title('uid iid inter')
  ax2.set_xlabel('uid')
  ax2.sharey(ax1)

  fig.savefig('{}/request_inter.png'.format(config.default.results_subdir))


  config.save()

