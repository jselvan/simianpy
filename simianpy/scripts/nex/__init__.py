import click

from .combine import Combine
from .from_raw import from_raw
from .info import Info


@click.group()
def Nex():
    pass


Nex.add_command(Combine)
Nex.add_command(Info)
Nex.add_command(from_raw)
