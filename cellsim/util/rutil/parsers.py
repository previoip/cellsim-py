import os.path
import requests
import mimetypes
import urllib.parse
from re import compile as re_compile

if not mimetypes.inited:
  mimetypes.init()

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

  @staticmethod
  def get_extension(resp: requests.Response):
    content_type = resp.headers.get('content-type', '')
    if not content_type:
      return ''
    if ';' in content_type:
      content_type = content_type.split(';')[0]
    if not mimetypes.inited:
      mimetypes.init()
    return mimetypes.guess_extension(content_type)

  @classmethod
  def get_filename(cls, resp: requests.Response) -> str:
    content_disposition = resp.headers.get('content-disposition', '')
    if not content_disposition:
      parsed_url = urllib.parse.urlparse(urllib.parse.unquote(str(resp.url).rstrip('\\/')))
      parsed_url = parsed_url._replace(fragment='')
      filename = os.path.basename(parsed_url.geturl())
    else:
      filename = cls._search_content_disposition_filename(content_disposition)
    filename = filename.strip('\'\\".;)*\ ')
    # RFC 6266 - attfn2231
    if filename.lower().startswith('utf-8'):
      filename = filename[7:]
    filename = cls.regexp_invalid_path_chars.sub('', filename)
    extension = cls.get_extension(resp)
    if extension and not filename.endswith(extension):
      filename += extension
    return filename

  @staticmethod
  def get_content_length(resp: requests.Response) -> int:
    cl = resp.headers.get('content-length', 0)
    if isinstance(cl, int):
      return cl
    if cl.isnumeric():
      return int(cl)
    return 0

  @classmethod
  def get_urlpath(cls, resp: requests.Response, include_netloc=False):
    parsed_url = urllib.parse.urlparse(urllib.parse.unquote(str(resp.url).rstrip('\\/')))
    parsed_url = parsed_url._replace(fragment='', query='')
    path = parsed_url.path.lstrip('\\/')
    path = cls.regexp_invalid_path_chars.sub('-', path)
    if include_netloc:
      path = os.path.join(parsed_url.netloc, path)
    return path

  @classmethod
  def get_urldir(cls, resp: requests.Response, include_netloc=False):
    return os.path.dirname(cls.get_urlpath(resp, include_netloc))

  @staticmethod
  def get_content_length(resp: requests.Response) -> int:
    cl = resp.headers.get('content-length', 0)
    if isinstance(cl, int):
      return cl
    if cl.isnumeric():
      return int(cl)
    return 0
