from .openephys import load
from ..nex import NexWriter, Nex
from ..File import File

from collections import defaultdict
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

class OpenEphys(File):
    """Interface for OpenEphys files

    Warnings
    --------
    This class is still under development!

    Handling of event data is currently hard coded into this class.
    Recommended use case is to subclass or monkey patch 

    Parameters
    ----------
    filename: str or Path 
    mode: str, optional, default: 'r' 
        Must be one of ['r']
    recipe: str, Path or list
        recipe describing what files to load
    start_time: pd.Timestamp, optional, default: 0
        If you wish to provide a specific start time,
        provide a pandas Timestamp using pd.to_datetime
    use_cache: bool, optional, default: False
        If True, data will be dumped into an HDF file via h5py.File
    cache_path: Path, file-like object or None, optional, default: None
        If a cache_path is provided, the HDF file will be saved.
        Any valid input to h5py.File is accepted
        If None, a temporary file (tempfile.TemporaryFile) will be 
        used. This file will be deleted upon closing the file
    overwrite_cache: bool, optional, default: False
        If false, data will not be loaded if already present in cache
        If true, data in cache will be overwritten
    logger: logging.Logger, optional
        logger for this object - see simi.io.File for more info
    
    Attributes
    ----------
    continuous_data
    spike_data
    event_data
    """
    description = """ """
    extension = ['.continuous', '.spikes', '.events']
    isdir = True
    needs_recipe = True
    default_mode = 'r'
    modes = ['r']

    def __init__(self, filename, **params):
        super().__init__(filename, **params)
        self.start_time = params.get('start_time', 0)

    def open(self):
        self._get_data_cache()
        if self.mode == 'r':
            for file_params in self.recipe:
                filename = file_params['file']
                filetype = file_params['type']
                varname = file_params['name']

                if not self.overwrite_cache and filetype in self._data.keys() and varname in self._data[filetype].keys():
                    continue
                if self.use_cache and filetype not in self._data.keys():
                    #unlike the defaultdict interface, h5py interface does not 
                    #tolerate missing labels unless we use h5py address syntax
                    self._data.create_group(filetype)

                fpath = Path(self.filename, filename)
                if not fpath.is_file():
                    raise FileNotFoundError(f"File ({fpath.name}) not found at {fpath.parent}")
                
                #header must be serialized to allow interoperability with hdf caching
                data = load(fpath, self.logger)
                header = data.pop('header')
                self._data[filetype][varname] = data
                self._data[filetype][varname]['header'] = json.dumps(header)

    def close(self):
        if self.mode == 'r':
            pass
    
    def write(self, filename):
        raise NotImplementedError

    @staticmethod
    def read_timestamps(timestamps, start):
        return pd.to_datetime(start, format = "%d-%b-%Y %H%M%S") + pd.to_timedelta(timestamps, unit = 's')

    def _parse_continuous_data(self, cnt_data):
        header = json.loads(cnt_data['header'])
        block_length = int(header['blockLength'])
        sampling_rate = int(header['sampleRate'])
        start_time = header['date_created']
        expanded_timestamps = ( np.expand_dims(cnt_data['timestamps'], axis = 1) \
            + np.expand_dims(np.arange(block_length)/sampling_rate, axis = 0) ).flatten()
        return pd.Series(
            cnt_data['data'],
            index = self.read_timestamps(
                timestamps=expanded_timestamps, 
                start=start_time
            )
        )

    def get_continuous_data(self, keys = None, resample_freq = None):
        """Get continuous data from openephys data as pandas dataframe

        Parameters
        ----------
        keys: list of str or None, optional, default: None
            subset of continuous data that will be retrieved
            if None, returns all data
        resample_freq: pd.DateOffset or str or None, optional, default: None
            valid time for new sample freq (e.g., '1L' or pd.offsets.Milli(1))
            if None, data is not resampled
        
        Returns
        -------
        continuous_data: pd.DataFrame
            columns will correspond to keys provided
            index will be a pd.DateTimeIndex using timestamps from openephys
        """
        if keys is None:
            keys = self._data['continuous'].keys()
        continuous_data = pd.DataFrame(
            {key: self._parse_continuous_data(self._data['continuous'][key]) for key in keys}
        )
        if resample_freq is not None:
            continuous_data = continuous_data.asfreq(resample_freq)
        return continuous_data
    
    def _parse_spike_data(self, spk_data):
        header = json.loads(spk_data['header'])
        sample_in_microseconds = f"{1e6/float(header['sampleRate']):.3f}U"
        start_time = header['date_created']
        return pd.DataFrame(
            spk_data['spikes'].squeeze(),
            columns = pd.timedelta_range(0, periods = spk_data['spikes'].shape[1], freq = sample_in_microseconds),
            index = pd.MultiIndex.from_arrays(
                [
                    spk_data['sortedId'].squeeze(),
                    self.read_timestamps(spk_data['timestamps'].squeeze(), start = start_time)
                ],
                names = ('Unit', 'Timestamp')
            )
        )

    def get_spike_data(self, keys = None):
        if keys is None:
            keys = self._data['spikes'].keys()
        spike_data = pd.concat(
            {key: self._parse_spike_data(self._data['spikes'][key]) for key in keys}
        )
        return spike_data
    
    def _parse_event_data(self, evt_data):
        header = json.loads(evt_data['header'])
        start_time = header['date_created']
        vars = ['timestamps', 'eventId', 'channel']
        evt_data_df = pd.DataFrame( {var: evt_data[var].squeeze() for var in vars} )
        evt_data_df['bitVal'] = 2**(7-evt_data_df['channel'])
        event_data = evt_data_df.query('eventId==1').groupby('timestamps').bitVal.sum()
        event_data.index = self.read_timestamps( timestamps=event_data.index , start=start_time )
        return event_data

    def get_event_data(self, keys = None):
        if keys is None:
            keys = self._data['events'].keys()
        event_data = pd.DataFrame(
            {key: self._parse_event_data(self._data['events'][key]) for key in keys}
        )
        return event_data

    def to_nex(self, nexfile_path, timestampFrequency, **params):
        #TODO implement to_nex function
        nexfile_path = Path(nexfile_path)
        
        start_time = time.time()
        with Nex(nexfile_path, mode='w', timestampFrequency=timestampFrequency, **params) as nexfile:
            pass

        self.logger.info(f'\nSuccessfully wrote nexfile at path: {nexfile_path}')
        self.logger.info(f'Total time: {(time.time() - start_time):.3f} seconds\n\n')
        return nexfile


    # unit_as_char = lambda x: chr(x - 1 + ord('a')) if x > 0 else 'U'
    # writer = nexfile.NexWriter(SamplingRate_spikes, useNumpy=True)

    # #add data
    # for i in range(num_channels):
    #     print("\nFor channel %d:" % (i + 1))
    #     #load spike data
    #     spike_fpath = os.path.join(ephys_path, f"{spike_prefix}.0n{i}.spikes")
    #     spike_data = load(spike_fpath)

    #     units = np.unique(spike_data['sortedId'])
    #     for unit_num, unit_id in enumerate(units):
    #         if spike_data['spikes'].shape[1] != NPointsWave:
    #             raise ValueError(f'The spikes file at the following path has the wrong number of NPointsWave. \n fpath: {spike_fpath}')

    #         unit_name = unit_as_char(unit_num)
    #         neuron_name = f"sig{i + 1:03d}{unit_name}"
    #         wave_name = f"{neuron_name}_wf"

    #         idx = (spike_data['sortedId'].squeeze() == unit_id)

    #         neuronTs = spike_data['timestamps'][idx]
    #         WaveformValues = spike_data['spikes'][idx]

    #         try:
    #             while WaveformValues.ndim > 3:
    #                 WaveformValues = WaveformValues.squeeze(axis=0)
    #         except:
    #             warnings.warn(f'Failed to shape WaveformValues appropriately. Skipping unit: {wave_name}')
    #             continue
            
    #         if WaveformValues.shape[1] != NPointsWave:
    #             warnings.warn(f'Waveforms for unit {wave_name} has {WaveformValues.shape[1]} points instead of {NPointsWave} points as specified by arg NPointsWave. NPointsWave will be adjusted for this unit - there may be unintended consequences.')

    #         #add neuron & spike waveforms
    #         writer.AddNeuron(name = neuron_name, timestamps = neuronTs)
    #         writer.AddWave(name = wave_name, 
    #         timestamps = neuronTs, 
    #         SamplingRate = SamplingRate_spikes, 
    #         WaveformValues = WaveformValues, 
    #         NPointsWave=NPointsWave,
    #         PrethresholdTimeInSeconds=PrethresholdTimeInSeconds,
    #         wire = channel,
    #         unit = unit_num
    #         )
        
    #     #add continuous data
    #     continuous_fpath = os.path.join(ephys_path, f"{LFP_prefix}{i + 1}.continuous")
    #     continuous_data = load(continuous_fpath)

    #     AD_name = f"AD{i + 1:02d}"

    #     #decimates by factor 30, using Chebyshev type I infinite impulse response filter of order 8 (in theory this is the same as MATLAB decimate)
    #     writer.AddContVarWithSingleFragment(name = AD_name,
    #     timestampOfFirstDataPoint = continuous_data['timestamps'][0],
    #     SamplingRate = SamplingRate_continuous,
    #     values = scipy.signal.decimate(continuous_data['data'], 30)
    #     )
    
    # #add eye channels
    # print("\nFor eye channel:")
    # for eye_channel, fname in eye_channels.items():
    #     continuous_fpath = os.path.join(ephys_path, fname)
    #     continuous_data = load(continuous_fpath)
        
    #     #decimates by factor 30, using Chebyshev type I infinite impulse response filter of order 8 (in theory this is the same as MATLAB decimate)
    #     writer.AddContVarWithSingleFragment(name = eye_channel,
    #     timestampOfFirstDataPoint = continuous_data['timestamps'][0],
    #     SamplingRate = SamplingRate_continuous,
    #     values = scipy.signal.decimate(continuous_data['data'], 30)
    #     )
    
    # #add event codes
    # print('\nFor events:')
    # event_fpath = os.path.join(ephys_path, 'all_channels.events')
    # event_data = load(event_fpath)

    # markers = []
    # timestamps = []
    # marker_val = 0
    # for timestamp, channel, eventId in zip(event_data['timestamps'], 2**(7 - event_data['channel']), event_data['eventId']):
    #     if eventId == 1:
    #         marker_val += channel
    #     else:
    #         if marker_val:
    #             markers.append(marker_val)
    #             timestamps.append(timestamp)
    #         marker_val = 0
    
    # writer.AddMarker(name = 'Strobed', timestamps = np.array(timestamps), fieldNames = np.array(['DIO']), markerFields = np.array([[f'{int(marker):03d}' for marker in markers]]))
    # writer.WriteNexFile(nexfile_path)