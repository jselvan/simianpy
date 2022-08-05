import click

from .to_nex import to_nex


@click.group()
def OpenEphys():
    pass


OpenEphys.add_command(to_nex)