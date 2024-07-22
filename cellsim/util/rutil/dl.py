import typing as t
import os.path
from io import RawIOBase
from requests import Session, Response, HTTPError
from .parsers import response_parser
from .util import new_tqdm_info_handler

request_header_default = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
  "Cache-Control": "no-cache",
  "Pragma": "no-cache"
}

def new_session():
  sess = Session()
  sess.headers.update(request_header_default)
  return sess

def stream_to_io(resp: Response, _io: RawIOBase, chunk_size=65536, stream_info_handler=new_tqdm_info_handler):
  content_length = response_parser.get_content_length(resp)
  handler = stream_info_handler(total=content_length)
  for chunk in resp.iter_content(chunk_size=chunk_size, decode_unicode=False):
    _io.write(chunk)
    handler.update(len(chunk))
  handler.refresh()


class request:

  @classmethod
  def _request(cls, sess: Session, method, url, /, **kwargs) -> Response:
    resp = sess.request(method, url, **kwargs)
    if resp.status_code == 200:
      return resp
    elif resp.status_code == 301 or resp.status_code == 302:
      return cls._request(sess, method, resp.url, **kwargs)
    else:
      resp.raise_for_status()
      raise HTTPError('{}: {} {} {}'.format(method, resp.url, resp.status_code, resp.reason))

  @classmethod
  def get(cls, sess, url, /, **kwargs) -> Response:
    return cls._request(sess, 'GET', url, **kwargs)

  @classmethod
  def head(cls, sess, url, /, **kwargs) -> Response:
    return cls._request(sess, 'HEAD', url, **kwargs)

  @classmethod
  def post(cls, sess, url, /, **kwargs) -> Response:
    return cls._request(sess, 'POST', url, **kwargs)


class downloader:

  @staticmethod
  def to_folder(sess, url, folder, stream_chunk_size=65536, stream_info_handler=new_tqdm_info_handler) -> str:
    resp = request.get(sess, url, allow_redirects=True, stream=True)
    filename = response_parser.get_filename(resp)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as fp:
      if not fp.writable():
        resp.close()
        raise OSError('unable to write to {}'.format(filepath))
      stream_to_io(resp, fp, chunk_size=stream_chunk_size, stream_info_handler=stream_info_handler)
    return filepath

  @staticmethod
  def to_io(sess, url, _io, stream_chunk_size=65536, stream_info_handler=new_tqdm_info_handler) -> Response:
    resp = request.get(sess, url, allow_redirects=True, stream=True)
    stream_to_io(resp, _io, chunk_size=stream_chunk_size, stream_info_handler=stream_info_handler)
    return resp
