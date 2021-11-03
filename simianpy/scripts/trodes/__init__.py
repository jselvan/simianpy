from .dump import dump
from .plot_channel import plot_channel
from .view import view

import click

@click.group()
def Trodes():
    pass

Trodes.add_command(dump)
Trodes.add_command(plot_channel)
Trodes.add_command(view)