from .nex import Nex
from .intan import Intan
from .trodes import Trodes
from .util import util

import click

@click.group()
def simi():
    pass

simi.add_command(Nex)
simi.add_command(Intan)
simi.add_command(Trodes)
simi.add_command(util)

if __name__ == '__main__':
    simi()