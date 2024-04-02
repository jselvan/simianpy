from pathlib import Path

import click

from simianpy.io.trodes.readtrodes import read_header


@click.command()
@click.argument("path")
def info(path):
    fieldsText, offset = read_header(path)
    print('Header Offset', offset)
    print(fieldsText)