from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np
import click

@click.command()
@click.argument('path')
@click.option('-c','--chunksize',default=1e7,help='max number of samples loaded \
into memory at once. tweak to improve performance. set to -1 to do entire file at once')
@click.option('--copy',default=False,is_flag=True,help='copies file before applying filter')
def cmr(path,chunksize,copy):
    """Apply common median reference to an npy file"""
    path = Path(path)
    if copy:
        filt_path = shutil.copy(path,path.parent/f"{path.stem}.cmr.npy")
    else:
        filt_path = path
    data = np.load(filt_path,mmap_mode='r+')
    if chunksize < 0:
        data -= np.median(data,axis=0,keepdims=True)
    else:
        for idx in tqdm(np.arange(0,data.shape[0],chunksize), desc='Applying median correction'):
            chunkslice = slice(int(idx), int(idx+chunksize))
            data[chunkslice,:] -= np.median(data[chunkslice,:],axis=1,keepdims=True).astype(int)