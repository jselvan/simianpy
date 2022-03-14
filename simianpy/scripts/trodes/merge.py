import shutil
from pathlib import Path

import click
import numpy as np
import yaml


@click.command()
@click.argument("input_directories", nargs=-1)
@click.option(
    "-o", "--output_directory", default=".", help="Output directory for merged files"
)
@click.option(
    "-r",
    "--recipe-path",
    required=True,
    help="Recipe specifying how to map Trodes data",
)
@click.option(
    "--remove-offset/--keep-offset",
    is_flag=True,
    default=True,
    help="Remove offset from timestamps in merged data",
)
def merge(input_directories, output_directory, recipe_path, remove_offset):
    """
    Merges trodes auxiliary files from multiple directories into a single directory.

    Note: This script will merge timestamps and event files.
    """
    input_directories = [Path(directory) for directory in input_directories]
    output_directory = Path(output_directory)
    recipe = yaml.safe_load(open(recipe_path))
    last_timestamp = 0

    # Check that all input directories exist
    if not all([directory.exists() for directory in input_directories]):
        raise ValueError("An input directory does not exist")

    # Initialize output data
    output_data = {}
    for name, info in recipe.items():
        if info["type"] == "analog":
            output_data[f"{name}.timestamps.npy"] = []
        elif info["type"] == "DIO":
            output_data[f"{name}.on.npy"] = []
            output_data[f"{name}.off.npy"] = []

    for directory in input_directories:
        ## get first timestamp
        for name, info in recipe.items():
            if info["type"] == "analog":
                timestamps = np.load(directory / f"{name}.timestamps.npy", "r")
                first_timestamp = timestamps[0]
                break
        else:
            raise ValueError("No analog data found in recipe")

        ## extract data from directory and adjust offsets
        for name, info in recipe.items():
            if info["type"] == "analog":
                ## copy channel names for first directory
                if not (output_directory / f"{name}.channels.txt").exists():
                    shutil.copy(directory / f"{name}.channels.txt", output_directory)

                ## load timestamps and adjust offset
                timestamps = np.load(directory / f"{name}.timestamps.npy")
                if remove_offset:
                    timestamps -= first_timestamp
                timestamps += last_timestamp
                output_data[f"{name}.timestamps.npy"].append(timestamps)
            elif info["type"] == "DIO":
                ## load on and off events and adjust offset
                for state in ["on", "off"]:
                    state_timestamps = np.load(directory / f"{name}.{state}.npy")
                    if remove_offset:
                        state_timestamps -= first_timestamp
                    state_timestamps += last_timestamp
                    output_data[f"{name}.{state}.npy"].append(state_timestamps)

        last_timestamp = timestamps[-1]

    ## save output data
    for file, data in output_data.items():
        data = np.concatenate(data)
        np.save(output_directory / file, data)
