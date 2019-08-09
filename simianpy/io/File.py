from ..misc import getLogger

import json
import yaml
from pathlib import Path

class File():
    """Base class for File IO"""
    description = """ """
    extension = ['']
    isdir = False
    needs_recipe = False
    modes = ['r']
    
    def __init__(self, filename, mode = 'r', **params):
        self.filename = Path(filename)
        assert mode in self.modes, f"Provided mode '{mode}' is not supported. Please provide one of: {self.modes}"
        self.mode = mode

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
            recipe_parsers = {'.json':json.load, '.yaml':yaml.safe_load}
            assert recipe.suffix in recipe_parsers.keys(), f"Provided recipe file format '{recipe.suffix}' not supported. Please provide one of {recipe_parsers.keys()}"
            with open(recipe, 'r') as f:
                self.recipe = recipe_parsers[recipe.suffix](f)
        else:
            raise TypeError('Provided recipe must be a list, dict or a Path object pointing to a recipe file')
    
    def __enter__(self):
        if not hasattr(self, 'open'):
            raise NotImplementedError('No open method exists for this class. Cannot use the context manager to interact.')
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.logger.warning(f"type={exc_type}\nvalue={exc_value}\ntraceback:\n{traceback}", exc_info=True)
        
        try:
            self.close()
        except NotImplementedError:
            self.logger.warning(f"The 'close' method is not implemented for this class. File closing may not be handled properly", exc_info=True)
            
    
    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
