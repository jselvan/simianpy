from collections import Counter
from pathlib import Path

import numpy as np

def load_raw(filename, shape, dtype='int16', mmap=True):
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"File not found: {filename}")

    filesize = filename.stat().st_size
    itemsize = np.dtype(dtype).itemsize
    nitems = filesize / itemsize

    if not nitems.is_integer():
        raise ValueError(f"Non integer number of items: {nitems}. File ({filename}): {filesize}; Itemsize: {itemsize}")


    if len(shape) == 1:
        raise NotImplementedError(f"Cannot handle shape of size 1")
    elif len(shape) == 2:
        pass
    else:
        raise NotImplementedError(f"Cannot handle shape of size {len(shape)}")

    if Counter(shape)[None] > 1:
        raise ValueError(f"Shape can only have 1 None: {shape}")
    shape = tuple(dim if dim is not None else nitems//np.prod(list(filter(lambda dim: dim is not None, shape))) for dim in shape)
    if all(isinstance(dim, int) or (hasattr(dim, 'is_integer') and dim.is_integer()) for dim in shape):
        shape = tuple(map(int, shape))
    else:
        raise ValueError(f"All dims must be integer values. shape={shape}")

    if mmap:
        data = np.memmap(filename=filename,dtype=dtype,shape=shape,mode='r')
    else:
        raise NotImplementedError(f"mmap=False not implemented")
        #TODO: implement load_raw without mmap 
        #data = np.fromfile(file=filename,dtype=dtype).reshape(shape)
    
    return data