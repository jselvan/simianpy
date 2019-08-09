from . import load_intan_rhs_format
from ..File import File

import numpy as np
import pandas as pd

def load(filename, **kwargs):
    allowed_extensions = ['.rhs']
    assert any(filename.endswith(ext) for ext in allowed_extensions), f"Provided filename '{filename}' ends with an invalid extension. Must be one of: {', '.join(allowed_extensions)}"
    if filename.endswith('rhs'):
        return load_intan_rhs_format.read_data(filename, **kwargs)

class RHS(File):
    """Interface for Intan RHS files ('rhs')

    Warnings
    --------
    This class is still under development!

    Parameters
    ----------
    filename: str or Path
    mode: str, optional, default: 'r'
        Must be one of ['r']
    recipe: str, Path or dict
        recipe describing how to read RHS file
    start_time: pd.Timestamp, optional, default: 0
        If you wish to provide a specific start time - provide a pandas Timestamp for the start time of the recording using pd.to_datetime
    notch: bool, optional, default: False
        Whether to use apply notch filter implemented by intan
    logger: logging.Logger, optional
        logger for this object - see simi.io.File for more info
    
    Attributes
    ----------
    continuous_data
    spike_data
    event_data
    stimulation_data
    """
    description = """ """
    extension = ['.rhs']
    isdir = False
    needs_recipe = True
    modes = ['r']

    def __init__(self, filename, **params):
        super().__init__(filename, **params)
        self.start_time = params.get('start_time', 0)
        self.notch = params.get('notch', False)

    def open(self):
        if self.mode == 'r':
            self._data = load_intan_rhs_format.read_data(self.filename, notch = self.notch, logger = self.logger)
    
    def close(self):
        pass

    def write(self, filename):
        raise NotImplementedError
    
    @property
    def start_time(self):
        return self._start_time
    
    @start_time.setter
    def start_time(self, start_time):
        self._start_time = pd.to_datetime(start_time)

    @property
    def timestamps(self):
        return pd.to_timedelta(self._data['t'], unit = 's') + self.start_time
    
    def get_continuous_data(self, keys = None):
        if keys is None:
            vars = self.recipe['continuous_data']
        else:
            keys = set(keys)
            vars = [var for var in self.recipe['continuous_data'] if var['name'] in keys]

            var_names = {var['name'] for var in vars}
            missing_keys = keys - var_names
            assert not missing_keys, f"The following keys are missing: {missing_keys}"

        continuous_data = pd.DataFrame(
            {var['name']: self._data[var['source']][var['idx']] for var in vars},
            index = self.timestamps
        )

        return continuous_data
    
    def get_event_data(self):
        def _get_events(eventinfo):
            eventdata = np.sum(
                [self._data[bit['source']][bit['idx']] * bit['bitval'] for bit in eventinfo], 
                axis = 0
            )
            event_idx = np.pad((eventdata[:-1] == 0) & (eventdata[1:] != 0), (1,0), 'constant')
            return pd.Series(eventdata[event_idx], index = self.timestamps[event_idx])

        return pd.DataFrame(
            {
                name: _get_events(eventinfo)
                for name, eventinfo in self.recipe['event_data'].items() 
            }
        )
    
    def get_stimulation_data(self):
        return pd.DataFrame(
            data = self._data['stim_data'].T, 
            index = self.timestamps, 
            columns = self.recipe['stimulation_data']
        )