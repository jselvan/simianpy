import click

from .info import Info


@click.group()
def Intan():
    pass


Intan.add_command(Info)
