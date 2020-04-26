from .combine import Combine
from .info import Info

import click

@click.group()
def Nex():
    pass

Nex.add_command(Combine)
Nex.add_command(Info)