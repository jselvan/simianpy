from .cmr import cmr
from .write_to_sound import write_to_sound

import click

@click.group()
def util():
    pass

util.add_command(cmr)
util.add_command(write_to_sound)