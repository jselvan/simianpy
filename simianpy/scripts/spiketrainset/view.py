from simianpy.analysis.spiketrain import SpikeTrainSet

import click

@click.command()
@click.argument('spiketrainset', type=click.Path(exists=True, dir_okay=False, readable=True))
def view(spiketrainset):
    """
    View a SpiketrainSet in a GUI.
    
    SPIKETRAINSET is the path to the SpikeTrainSet file.
    """
    spiketrain = SpikeTrainSet.load(spiketrainset)
    spiketrain.view()