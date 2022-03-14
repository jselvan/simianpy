from pathlib import Path

import click

from simianpy.io.trodes import Trodes, infer_session_name


@click.command()
@click.argument("path")
@click.option(
    "-s",
    "--session-name",
    default=None,
    help="provide session name if it can't be inferred",
)
@click.option("-o", "--output", default=None, help="output directory. defaults to PATH")
@click.option(
    "-c",
    "--chunksize",
    default=1e7,
    help="max number of samples loaded into memory at once. tweak to improve performance",
)
@click.option(
    "-r", "--recipe-path", default=None, help="Recipe specifying how to map Trodes data"
)
@click.option("-v", "--verbose", default=False, is_flag=True)
@click.option(
    "-e",
    "--end",
    default=None,
    help="Truncate to this timestamp (in samples) if provided",
)
def dump(path, session_name, output, chunksize, recipe_path, verbose, end):
    path = Path(path)
    session_name = infer_session_name(path) if session_name is None else session_name
    if recipe_path is None:
        raise ValueError("Must specify path to recipe")
    output = path if output is None else Path(output)
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
        trodes.dump(output, chunksize, end=end)
