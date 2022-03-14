import shutil
from pathlib import Path

import click
import numpy as np
import numpy.lib.format as fmt


def get_header(files, axis):
    """get header for concatenated file

    Parameters
    ----------
    files : path-like
        npy files to concatenate
    axis : int, default=0
        axis to cocatenate along

    Returns
    -------
    dict
        datatype descr, fortran order and shape, for concatenated file
    """
    dtypes, shapes, forders = [], [], []
    for file in files:
        f = np.load(file, "r")
        dtypes.append(f.dtype)
        shapes.append(f.shape)
        forders.append(f.flags["F_CONTIGUOUS"])

    if all(dtype == dtypes[0] for dtype in dtypes):
        dtype = dtypes[0]
    else:
        raise ValueError("All files must have the same dtype")

    if all(forder == forders[0] for forder in forders):
        forder = forders[0]
    else:
        raise ValueError("All files must have the same fortran order")

    if all(len(shape) == len(shapes[0]) for shape in shapes):
        ndims = len(shapes[0])
    else:
        raise ValueError("All files must have the same number of dimensions")

    if all(
        all(shape[axis_] == shapes[0][axis_] for shape in shapes)
        for axis_ in range(ndims)
        if axis_ != axis
    ):
        shape = list(shapes[0])
        shape[axis] = sum(shape_[axis] for shape_ in shapes)
        shape = tuple(shape)
    else:
        raise ValueError(
            "All files must have the same shape along the concatenation axis"
        )

    header = {
        "descr": fmt.dtype_to_descr(dtype),
        "fortran_order": forder,
        "shape": shape,
    }
    return header


@click.command("concat")
@click.argument("files", nargs=-1)
@click.option("-o", "--output", default="concat.bin")
@click.option("-f", "--force", is_flag=True, default=False)
@click.option("-t", "--type", default="raw", type=click.Choice(["raw", "npy"]))
@click.option("-a", "--axis", default=0, type=int)
def concat(files, output, force, type, axis):
    output = Path(output)
    files = [Path(file) for file in files]
    if any(not file.is_file() for file in files):
        raise FileNotFoundError("One or more files not found")
    if output.is_file() and not force:
        raise FileExistsError("Use -f to overwrite existing file")

    print("Concatenating")
    for idx, file in enumerate(files):
        print("\t", idx + 1, file.name)
    print("Into", "\n\t", output.name)

    with open(output, "wb") as outputfile:
        if type == "npy":
            fmt.write_array_header_2_0(outputfile, get_header(files, axis))
        for file in files:
            with open(file, "rb") as inputfile:
                if type == "npy":
                    inputfile.seek(128)
                shutil.copyfileobj(inputfile, outputfile)

    print("Concatenation complete")
