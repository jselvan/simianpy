from .nex import Nex
from .intan import Intan
from .trodes import Trodes

import click

@click.group()
def simi():
    pass

simi.add_command(Nex)
simi.add_command(Intan)
simi.add_command(Trodes)

if __name__ == '__main__':
    simi()