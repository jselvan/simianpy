from .combine import Combine

import click

@click.group()
def Nex():
    pass

Nex.add_command(Combine)