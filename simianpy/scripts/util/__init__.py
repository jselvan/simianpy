import click

from .cmr import cmr
from .concat import concat
from .h5tree import h5tree
from .write_to_sound import write_to_sound


@click.group()
def util():
    pass


util.add_command(cmr)
util.add_command(write_to_sound)
util.add_command(concat)
util.add_command(h5tree)
