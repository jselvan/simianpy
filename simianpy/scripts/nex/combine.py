from pathlib import Path

import click
from tqdm import tqdm

import simianpy as simi


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "-o",
    "--output",
    "output",
    default="out.nex",
    help="Output filepath. Default=out.nex",
)
@click.option(
    "-?",
    "--whatif",
    "whatif",
    default=False,
    help="Prompts user before any actions are taken",
    is_flag=True,
)
@click.option(
    "-f",
    "--force",
    "force",
    default=False,
    help="Overwrite files if necessary",
    is_flag=True,
)
def Combine(whatif, force, inputs, output):
    """ Combines multiples nex files into one

    INPUTS, nex files to be combined
    """
    nex_files = []
    for filepath in inputs:
        nex_files.extend(Path().glob(filepath))
    output = Path(output)

    if output.is_file() and not force:
        raise FileExistsError(
            f"A nex file named {output} already exists. Specify a different output file name or use the force parameter to overwrite."
        )

    if whatif:
        print("Merging:")
        print("\t-", "\n\t- ".join(map(str, nex_files)))
        print("Into:")
        print("\t", output)
        click.confirm("Do you want to proceed?", abort=True)

    variable_data = []
    timestampFrequencies = []
    for nex_file in tqdm(nex_files, "Reading files"):
        with simi.io.Nex(nex_file) as in_file:
            timestampFrequencies.append(in_file.data["FileHeader"]["Frequency"])
            variable_data.extend(in_file.data["Variables"])

    max_timestampFrequency = max(timestampFrequencies)
    if any(frequency != max_timestampFrequency for frequency in timestampFrequencies):
        pass  # should we raise error here or just a warning?

    print(f"Writing to {output}...")
    with simi.io.Nex(
        output, mode="w", timestampFrequency=max_timestampFrequency
    ) as out_file:
        out_file.data["Variables"] = variable_data

    print("Done!")
