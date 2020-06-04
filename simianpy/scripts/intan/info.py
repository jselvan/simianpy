from simianpy.io.intan.intanutil.read_header import read_header
from simianpy.io.intan.intanutil.get_bytes_per_data_block import get_bytes_per_data_block

from pathlib import Path
import os

import yaml
import click

@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('file', required=True, type=click.Path(exists=True,dir_okay=False))
@click.option('--showheader', is_flag=True, default=False)
def Info(file, showheader):
    filesize = os.path.getsize(file)
    with open(file, 'rb') as intan:
        header = read_header(intan)
        bytes_remaining = filesize - intan.tell()
    click.echo(f"Filename: {file}")
    if showheader:
        click.echo("\nHeader:\n")
        click.echo(yaml.dump(header))

    bytes_per_block = get_bytes_per_data_block(header)
    click.echo(f"\nAmplifiers were sampled at {header['sample_rate']/1e3:0.2f} kS/s.")
    if bytes_remaining == 0:
        click.echo("File contains no data.")
    elif bytes_remaining % bytes_per_block != 0:
        click.echo("File corrupt - number of data blocks is not whole")
    else:        
        num_data_blocks = int(bytes_remaining / bytes_per_block) 
        record_time = 128 * num_data_blocks / header['sample_rate']
        click.echo(f"File contains {record_time:0.3f} seconds of data.")