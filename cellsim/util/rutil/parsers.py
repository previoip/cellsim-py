import os.path
import requests
import urllib.parse
from re import compile as re_compile

class response_parser:

  regexp_headers_filename = re_compile(r'filename=(.+)')
  regexp_headers_filename_wildcard = re_compile(r'filename\*=(.+)')
  regexp_invalid_path_chars = re_compile(r'[\\\/\>\<:"\|\?\*%\x00-\x1f]+')

  @classmethod
  def _search_content_disposition_filename(cls, s) -> str:
    se1 = cls.regexp_headers_filename.search(s)
    if se1:
      return se1.group()
    se2 = cls.regexp_headers_filename_wildcard.search(s)
    if se2:
      return se2.group()
    return s



  @classmethod
  def get_filename(cls, resp: requests.Response) -> str:
    content_disposition = resp.headers.get('content-disposition', '')
    eval_default = lambda: os.path.basename(urllib.parse.unquote(resp.url.rstrip('\\/')))
    if not content_disposition:
      return eval_default()
    filename = cls._search_content_disposition_filename(content_disposition)
    if not filename:
      return eval_default()
    filename = filename.strip('\'\\".;)*\ ')
    # RFC 6266 - attfn2231
    if filename.lower().startswith('utf-8'):
      filename = filename[7:]
    filename = cls.regexp_invalid_path_chars.sub('', filename)
    return filename

  @staticmethod
  def get_content_length(resp: requests.Response) -> int:
    cl = resp.headers.get('content-length', 0)
    if isinstance(cl, int):
      return cl
    if cl.isnumeric():
      return int(cl)
    return 0
