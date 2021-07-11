from .combine import Combine
from .info import Info
from .from_raw import from_raw

import click

@click.group()
def Nex():
    pass

Nex.add_command(Combine)
Nex.add_command(Info)
Nex.add_command(from_raw)