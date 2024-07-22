import tqdm

def new_tqdm_info_handler(total=0, **kwargs) -> tqdm.tqdm:
  base_kwargs = {
      'total': total,
      'unit_scale': True,
      'unit_divisor': 1000,
      'unit': 'b',
  }
  kwargs.update(base_kwargs)
  return tqdm.tqdm(**kwargs)
