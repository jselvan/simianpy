from ..misc import getLogger

import json
from pathlib import Path

class File():
    """Base class for File IO"""
    description = """ """
    extension = ['']
    isdir = False
    needs_recipe = False
    modes = ['r']
    
    def __init__(self, filename, **params):
        self.filename = Path(filename)

        if 'logger' in params:
            self.logger = params['logger']
        else:
            logger_kwargs = params.get('logger_kwargs', {})
            logger_defaults = {'loggerName': __name__, 'fileName': self.filename.with_suffix('.log')}
            for k, v in logger_defaults.items():
                if k not in logger_kwargs:
                    logger_kwargs[k] = v

            self.logger = getLogger(**logger_kwargs)
        
        if self.needs_recipe:
            self._read_recipe(params['recipe'])
        else:
            self.recipe = None

    def _read_recipe(self, recipe):
        if isinstance(recipe, list) or isinstance(recipe, dict):
            self.recipe = recipe
        elif isinstance(recipe, Path):
            assert recipe.is_file(), f"Cannot find file at path: {recipe}"
            assert recipe.suffix == '.json', f"Provided file must a .json file not: {recipe}"
            with open(Path(recipe), 'r') as f:
                self.recipe = json.load(f)
        else:
            raise TypeError('Provided recipe must be a list, dict or a Path object pointing to a json file')