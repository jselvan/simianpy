from .nex import Nex

import click

@click.group()
def simi():
    pass

simi.add_command(Nex)

if __name__ == '__main__':
    simi()