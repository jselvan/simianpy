from .cmr import cmr

import click

@click.group()
def util():
    pass

util.add_command(cmr)