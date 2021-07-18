from .dump import dump

import click

@click.group()
def Trodes():
    pass

Trodes.add_command(dump)