import simianpy as simi

from pathlib import Path

from tqdm import tqdm
import numpy as np
import click

@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('rawpath')
@click.option('-s', '--shape', 'shape', default=None, help='ex: (1000, 128) provide shape as tuple for non-npy files')
@click.option('-t', '--transpose', 'transpose', default=False, is_flag=True, help='if data shape (N_CHANNEL, N_SAMPLE)')
@click.option('-o','--output','output', default='out.nex5', help='Output filepath. Default=out.nex')
@click.option('-f', '--force', 'force', default=False, help='Overwrite files if necessary', is_flag=True)
@click.option('-sf','--sampling-frequency', 'sampling_freq', default=3e4)
@click.option('-c','--channels','channels',default=None,help='Channel list file. Text file with N_CHANNEL lines')
@click.option('-m','--memmap','memmap', default=False,is_flag=True,help='Use memory mapping')
def from_raw(rawpath, shape, transpose, output, force, sampling_freq, channels, memmap):
    rawpath, output = Path(rawpath), Path(output)
    if output.is_file() and not force:
        raise FileExistsError("File already exists at" + output.as_posix())
    if rawpath.suffix == '.npy':
        raw = np.load(rawpath,mmap_mode='r' if memmap else None)
    else:
        raise NotImplementedError('Only .npy files are supported at the moment')
    if not transpose:
        raw = raw.T
    n_channels = raw.shape[0]
    if channels is None:
        channel_names = [f"AD{i}" for i in range(n_channels)]
    else:
        channel_names = open(channels,'r').read().splitlines()
    with simi.io.Nex(output, mode='w', timestampFrequency=sampling_freq) as nex:
        for name, row in tqdm(zip(channel_names, raw), total=n_channels):
            nex.writer.AddContVarWithSingleFragment(name, 0, sampling_freq, row)
        print('Writing...')
    print('Done!')