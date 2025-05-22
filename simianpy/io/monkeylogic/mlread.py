from typing import Optional, Literal, Dict, Iterator, Any
from pathlib import Path
from os import PathLike
import warnings

from simianpy.misc.logging import getLogger
from simianpy.io.monkeylogic.bhv2 import read_bhv2
from simianpy.io.monkeylogic.h5 import read_h5


def load(
    filepath: PathLike,
    logger=None,
    loader: Optional[Literal[".bhv2", ".h5"]] = None,
    include_user_vars: bool = True,
    pbar: bool = True,
) -> Iterator[Dict[str, Any]]:
    if logger is None:
        logger = getLogger(__name__)
    else:
        warnings.warn("Logger provided, but currently these functions do not use it")

    filepath = Path(filepath)
    filetypes = {
        ".bhv2": read_bhv2,
        ".h5": read_h5,
        ".bhv": NotImplemented,
        ".mat": NotImplemented,
    }
    if loader is None:
        load_fun = filetypes.get(filepath.suffix, NotImplemented)
    else:
        load_fun = filetypes.get(loader, NotImplemented)

    assert load_fun is not NotImplemented, NotImplementedError(
        f'Files with extension "{filepath.suffix}" are not yet supported'
    )

    return load_fun(filepath, logger=logger, include_user_vars=include_user_vars, pbar=pbar)
