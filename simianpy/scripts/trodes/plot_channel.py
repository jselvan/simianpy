import click
import matplotlib.pyplot as plt

from simianpy.io.trodes.readtrodes import readTrodesExtractedDataFile


@click.command()
@click.argument("path")
def plot_channel(path):
    _, data = readTrodesExtractedDataFile(path)
    plt.plot(data["voltage"])
    plt.show()
