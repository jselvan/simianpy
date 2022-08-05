from pathlib import Path

import click

from simianpy.io import OpenEphys


@click.command()
@click.argument("openephys_path", type=click.Path(exists=True))
@click.argument("recipe_path", type=click.Path(exists=True))
@click.option("-o", "--output-path", type=click.Path(), default=None)
@click.option("-f", "--force", is_flag=True, default=False)
@click.option("--sampling-rate", type=float, default=30000)
@click.option("-v", "--verbose", count=True)
def to_nex(openephys_path, recipe_path, output_path, force, sampling_rate, verbose):
    if output_path is None:
        output_path = Path(openephys_path).with_suffix(".nex5")
    else:
        output_path = Path(output_path)
    if output_path.is_file() and not force:
        raise FileExistsError(
            f"output file already exists: {output_path}. Use force parameter to overwrite."
        )
    if verbose > 1:
        loglevel = "DEBUG"
    elif verbose > 0:
        loglevel = "INFO"
    else:
        loglevel = "WARN"
    openephys = OpenEphys(
        openephys_path,
        time_units="s",
        recipe_path=recipe_path,
        logger_kwargs=dict(printLevel=loglevel),
    )
    with openephys as file:
        file.to_nex(output_path, sampling_rate, overwrite=force)
