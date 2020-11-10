from pathlib import Path
from ...misc import getLogger

def load(filepath, logger = None):
    if logger is None:
        logger = getLogger(__name__)
    
    filepath = Path(filepath)
    filetypes = {
        '.bhv2': NotImplemented,
        '.h5': NotImplemented,
        '.mat': NotImplemented,
        '.bhv': NotImplemented
    }
    
    load_fun = filetypes.get(filepath.suffix, NotImplemented)

    assert load_fun is not NotImplemented, NotImplementedError(f'Files with extension "{filepath.suffix}" are not yet supported')

    return load_fun(filepath, logger)