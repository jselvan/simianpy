from .info import Info

import click

@click.group()
def Intan():
    pass

Intan.add_command(Info)