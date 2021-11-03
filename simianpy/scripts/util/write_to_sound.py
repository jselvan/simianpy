import simianpy as simi

from pathlib import Path

import click

@click.command()
@click.argument('input')
@click.argument('channel')
@click.option('-n','--nchannels',default=385,type=int)
@click.option('-o','--output', default=None)
@click.option('-s','--samples', default=100000, type=int)
@click.option('-r','--sampling-rate', default=30000, type=int)
def write_to_sound(input, channel, nchannels, output, samples, sampling_rate):
    import wavio
    input_path, output_path = Path(input), Path(output)
    raw = simi.io.load_raw(input_path, (None,nchannels))[:samples,channel]
    wavio.write(output_path,raw,int(sampling_rate))