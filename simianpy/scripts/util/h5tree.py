from pathlib import Path

import click
import h5py

from simianpy.misc.tree import Tree


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-md", "--max-depth", default=None, type=int)
@click.option("-mc", "--max-children", default=None, type=int)
@click.option("-s", "--sort", "sort_", default=False, is_flag=True)
def h5tree(path, max_depth, max_children, sort_):
    path = Path(path)
    print("Tree for", path.name)
    with h5py.File(path, "r") as f:
        tree = Tree.from_hdf5("root", f)
    tree.print(maxdepth=max_depth, maxchildren=max_children, sortnodes=sort_)
