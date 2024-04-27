import click

from .dump import dump
from .merge import merge
from .plot_channel import plot_channel
from .view import view
from .info import info

@click.group()
def Trodes():
    pass


Trodes.add_command(dump)
Trodes.add_command(plot_channel)
Trodes.add_command(view)
Trodes.add_command(merge)
Trodes.add_command(info)