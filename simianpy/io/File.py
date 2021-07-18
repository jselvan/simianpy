from simianpy.misc import getLogger

import json
import yaml
from pathlib import Path
from tempfile import TemporaryFile
from collections import defaultdict

import h5py

class File():
    """Base class for File IO"""
    description = """ """
    extension = ['']
    isdir = False
    needs_recipe = False
    default_mode = 'r'
    modes = ['r']
    
    def __init__(self, filename, **params):
        self.filename = Path(filename)
        
        self.mode = params.get('mode', self.default_mode)
        if self.mode not in self.modes:
            raise ValueError(f"Provided mode '{self.mode}' is not supported. Please provide one of: {self.modes}")

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
            if 'recipe' in params:
                self.recipe = params['recipe'] 
            elif 'recipe_path' in params:
                self.recipe = self._read_recipe(params['recipe_path'])
            else:
                raise ValueError(f"If 'needs_recipe', must provide one of 'recipe' or 'recipe_path'. Provided params: {params}")
        else:
            self.recipe = None

        self.use_cache = params.get('use_cache', False)
        self.cache_path = params.get('cache_path', None)
        self.overwrite_cache = params.get('overwrite_cache', False)
        if not (self.cache_path is None or self.use_cache):
            raise ValueError(f"cannot provide cache_path if use_cache is not True")
    
    def _get_data_cache(self):
        self._data = h5py.File(self.cache_path or TemporaryFile()) if self.use_cache else defaultdict(dict)
    
    def _close_data_cache(self):
        if self.use_cache:
            if hasattr(self, '_data'):
                self._data.close()
        if hasattr(self, '_data'):
            del self._data
        
    def _read_recipe(self, recipe_path):
        recipe_path = Path(recipe_path)
        if not recipe_path.is_file():
            raise FileNotFoundError(f"Cannot find file at path: {recipe_path}")
        recipe_parsers = {'.json':json.load, '.yaml':yaml.safe_load}
        if recipe_path.suffix not in recipe_parsers.keys():
            raise ValueError(f"Provided recipe file format '{recipe_path.suffix}' not supported. Please provide one of {recipe_parsers.keys()}")
        with open(recipe_path, 'r') as f:
            recipe = recipe_parsers[recipe_path.suffix](f)
        return recipe
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.logger.error(f"type={exc_type}\nvalue={exc_value}\ntraceback:\n{traceback}", exc_info=True)
        self._close_data_cache()
        try:
            self.close()
        except NotImplementedError:
            self.logger.warning(f"The 'close' method is not implemented for this class. File closing may not be handled properly", exc_info=True)
            
    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
