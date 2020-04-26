import simianpy as simi

from pathlib import Path

from tqdm import tqdm
import click

@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('file', required=True, type=click.Path(exists=True,dir_okay=False))
@click.option('--expandvars', is_flag=True, default=False)
def Info(file, expandvars):
    header = simi.io.nex.read_header(file)
    click.echo(f"Filename: {file}")
    click.echo("\nHeader:")
    click.echo(header['FileHeader'])

    click.echo("\nVariables:")

    for var_idx, var in enumerate(header['Variables']):
        var_type = simi.io.Nex.vartypes_dict[var['Header']['Type']]
        var_name = var['Header']['Name']
        click.echo(f"[{var_idx}] {var_type.upper()} {var_name}")
        if expandvars:
            click.echo(var['Header'])
    