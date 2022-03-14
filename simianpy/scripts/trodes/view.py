from pathlib import Path

import click

from simianpy.io.trodes import Trodes, infer_session_name
from simianpy.plotting.scrolling_viewer import ScrollingViewerMultiChannel


@click.command()
@click.argument("path")
@click.option(
    "-s",
    "--session-name",
    default=None,
    help="provide session name if it can't be inferred",
)
@click.option(
    "-r", "--recipe-path", default=None, help="Recipe specifying how to map Trodes data"
)
@click.option("-v", "--verbose", default=False, is_flag=True)
def view(path, session_name, recipe_path, verbose):
    path = Path(path)
    session_name = infer_session_name(path) if session_name is None else session_name
    if recipe_path is None:
        raise ValueError("Must specify path to recipe")
    printLevel = "DEBUG" if verbose else "WARN"

    kwargs = dict(
        filename=path,
        session_name=session_name,
        recipe_path=recipe_path,
        mmap=True,
        pbar=True,
        logger_kwargs=dict(printLevel=printLevel),
    )
    with Trodes(**kwargs) as trodes:
        data = list(trodes._data["raw"]["data"].values())
        ScrollingViewerMultiChannel(data, cmr=True).show()


#  python -m simianpy.scripts trodes view D:\20210709_101444 -r C:\Users\selja\OneDrive\Research\Code\simianpy\test\recipe.yaml
