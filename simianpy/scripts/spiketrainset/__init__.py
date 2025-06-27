import click

from simianpy.scripts.spiketrainset.view import view

@click.group()
def SpikeTrain():
    pass

SpikeTrain.add_command(view)