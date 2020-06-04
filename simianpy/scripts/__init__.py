from .nex import Nex
from .intan import Intan

import click

@click.group()
def simi():
    pass

simi.add_command(Nex)
simi.add_command(Intan)

if __name__ == '__main__':
    simi()