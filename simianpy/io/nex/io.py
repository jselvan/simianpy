from .nexfile import Reader, NexWriter
from ..File import File

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

def load(filename, useNumpy = True):
    allowed_extensions = ['.nex', '.nex5']
    filename = Path(filename)
    if not filename.suffix in allowed_extensions:
        raise ValueError(f"Provided filename '{filename}' ends with an invalid extension. Must be one of: {', '.join(allowed_extensions)}")
    reader = Reader(useNumpy=useNumpy)
    return reader.ReadNexFile(filename)

def read_header(filename):
    reader = Reader()
    header = reader.ReadHeader(filename)
    return header

class Nex(File):
    """Interface for Neuroexplorer files ('.nex', '.nex5')

    Parameters
    ----------
    filename: str or Path
    mode: str, optional, default: 'r+'
        Must be one of ['r', 'r+','w']
    start_time: pd.Timestamp, optional, default: 0
        If you wish to provide a specific start time - provide a pandas Timestamp for the start time of the recording using pd.to_datetime
    useNumpy: bool, optional, default: True
        If True, use numpy arrays to hold data (recommended use)
    timestampFrequency: int, optional, default: None
        If mode = 'w', timestampFrequency must be provided (in Hz). Else will be inferred from file.
    logger: logging.Logger, optional
        logger for this object - see simi.io.File for more info
    
    Attributes
    ----------
    vartypes_dic
    vartypes_dict_rev

    data
    vartypes
    varnames
    fileLength
    start_time

    continuous_data
    spike_data
    event_data
    """
    description = """ """
    extension = ['.nex', '.nex5']
    default_mode = 'r+'
    modes = ['r', 'r+', 'w']
    isdir = False
    vartypes_dict = {
        0: 'neuron',
        1: 'events',
        2: 'interval',
        3: 'waveforms',
        4: 'population_vector',
        5: 'continuous',
        6: 'markers'
    }
    vartypes_dict_rev = {v:k for k,v in vartypes_dict.items()}

    def __init__(self, filename, **params):
        super().__init__(filename, **params)
        self.writer = None
        self.start_time = params.get('start_time', 0)
        self.useNumpy = params.get('useNumpy', True)
        self.timestampFrequency = params.get('timestampFrequency', None)

    def open(self):
        if self.mode in ['r','r+']:
            data = load(self.filename, useNumpy = self.useNumpy)
            if self.timestampFrequency is None:
                self.timestampFrequency = data['FileHeader']['Frequency']
            else:
                if self.timestampFrequency != data['FileHeader']['Frequency']:
                    self.logger.warning(f"provided timestampFrequency {self.timestampFrequency} does not match file header {data['FileHeader']['Frequency']}")
            self.writer = NexWriter(timestampFrequency = self.timestampFrequency, useNumpy = self.useNumpy)
            self.writer.fileData = data
        elif self.mode == 'w':
            if self.timestampFrequency is None:
                raise ValueError(f'You must provide a timestampFrequency if mode = "w"')
            self.writer = NexWriter(timestampFrequency = self.timestampFrequency, useNumpy = self.useNumpy)
    
    def close(self):
        if self.mode in ['r+', 'w']:
            self.write()

    def write(self, filename = None, saveContValuesAsFloats = 0):
        if filename is None:
            filename = self.filename
        else:
            filename = Path(filename)
        
        if filename.suffix == '.nex':
            self.writer.WriteNexFile(filename)
        elif filename.suffix == '.nex5':
            self.writer.WriteNex5File(filename, saveContValuesAsFloats = saveContValuesAsFloats)

    @property
    def data(self):
        if self.writer is None:
            raise AttributeError('File has not been opened yet!')
        else:
            return self.writer.fileData
    
    @data.setter
    def data(self, data):
        if self.writer is None:
            raise AttributeError('File has not been opened yet!')
        else:
            self.writer.fileData = data
    
    @property
    def vartypes(self):
        return np.array([ self.vartypes_dict[ var['Header']['Type'] ] for var in self.data['Variables']])

    @property
    def varnames(self):
        return np.array([var['Header']['Name'] for var in self.data['Variables']])

    @property
    def _vararray(self):
        return np.array(self.data['Variables'])
    
    @property
    def fileLength(self):
        return self.data['FileHeader']['End'] - self.data['FileHeader']['Beg'] 
    
    @property
    def start_time(self):
        return self._start_time
    
    @start_time.setter
    def start_time(self, start_time):
        self._start_time = pd.to_datetime(start_time)

    def _get_timestamps(self, timestamps):
        return pd.to_datetime(timestamps, unit='s', origin=self.start_time)

    def _get_continuous_data(self, var):
        assert var['Header']['Type'] == self.vartypes_dict_rev['continuous'], f'Must be a continuous variable'
        return pd.concat([
            pd.Series(var['ContinuousValues'][idx:idx+count] , index=pd.date_range(timestamp, periods = count, freq = f"{1e3/var['Header']['SamplingRate']:.05f}L")) 
            for timestamp, idx, count in zip(self._get_timestamps(var['Timestamps']), var['FragmentIndexes'], var['FragmentCounts'])
            ])

    def get_continuous_data(self):
        return pd.DataFrame({var['Header']['Name']: self._get_continuous_data(var) for var in self._vararray[self.vartypes == 'continuous']})

    def get_spike_data(self):
        return pd.concat(
            [
                pd.DataFrame(
                    var['WaveformValues'],
                    columns = pd.timedelta_range(0, periods = var['Header']['NPointsWave'], freq = f"{1e6/var['Header']['SamplingRate']:.3f}U"),
                    index = pd.MultiIndex.from_arrays(
                        [
                            [var['Header']['Wire']]*var['Timestamps'].size,
                            [var['Header']['Unit']]*var['Timestamps'].size, 
                            self._get_timestamps(var['Timestamps'])
                        ],
                        names = ('Channel', 'Unit', 'Timestamps')
                    )
                )
                for var in self._vararray[self.vartypes == 'waveforms']
            ]
        )

    def get_event_data(self):
        return pd.DataFrame({
            f"{var['Header']['Name']}/{name}":pd.Series(markers, index = self._get_timestamps(var['Timestamps'])) 
            for var in self._vararray[self.vartypes == 'markers']
            for name, markers in zip(var['MarkerFieldNames'], var['Markers']) 
        })