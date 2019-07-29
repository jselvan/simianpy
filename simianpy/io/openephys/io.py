from .openephys import load
from ..nex import NexWriter, Nex
from ..File import File

from collections import defaultdict
from pathlib import Path
import time

import pandas as pd

class OpenEphys(File):
    """Interface for OpenEphys files

    Warnings
    --------
    This class is still under development!

    Parameters
    ----------
    filename: str or Path
    recipe: str, Path or list
        recipe describing what files to load
    start_time: pd.Timestamp, optional, default: 0
        If you wish to provide a specific start time - provide a pandas Timestamp for the start time of the recording using pd.to_datetime
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
    modes = ['r', 'rp']

    def open(self, mode = 'r'):
        assert mode in self.modes, f'Mode ({mode}) not supported. Try: {self.modes}'
        self._data = defaultdict(dict)
        if mode == 'r':
            for file_params in self.recipe:
                fpath = Path(self.filename, file_params['file'])
                assert fpath.is_file(), f"File ({fpath.name}) not found at {fpath.parent}"
                self._data[file_params['type']][file_params['name']] = load(fpath, self.logger)
        elif mode == 'rp':
            pass

    
    def write(self, filename):
        raise NotImplementedError

    @staticmethod
    def read_timestamps(timestamps, start):
        return pd.to_datetime(start, format = "%d-%b-%Y %I%M%S") + pd.to_timedelta(timestamps, unit = 's')

    def _parse_continuous_data(self, var_data):
        return pd.Series(
            var_data['data'],
            index = self.read_timestamps(
                timestamps=var_data['timestamps'], 
                start=var_data['header'][' date_created'].strip("'")
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
    
    def _parse_spike_data(self, var_data):
        return pd.DataFrame(
            var_data['spikes'],
            columns = pd.timedelta_range(0, periods = var_data['spikes'].shape[1], freq = f"{1e6/var_data['header']['sampleRate']:.3f}U"),
            index = pd.MultiIndex.from_arrays(
                [
                    [var_data['Header']['Unit']]*var_data['Timestamps'].size, 
                    var_data['sortedId'],
                    self.read_timestamps(var_data['Timestamps'], start = var_data['header'][' date_created'].strip("'"))
                ],
                names = ('Channel', 'Unit', 'Timestamps')
            )
        )

    def get_spike_data(self, keys = None):
        return pd.concat(
            [
                self._parse_spike_data(var)
                for name, var in self._data['spikes'].items()
            ]
        )


        # return pd.DataFrame(
        #     {name: pd.Series(
        #         var['data']['spikes'], 
        #         index = self.read_timestamps(
        #             timestamps = var['data']['timestamps'], 
        #             start = var['data']['header'][' date_created'].strip("'")
        #             )
        #         ) 
        #         for name, var in self._data['spikes'].items()
        #     })
    
        # pd.DataFrame(d['spikes'][:,:,0], index = self.read_timestamps(d['timestamps'], d['header'][' date_created'].strip("'")))
    
    @property
    def event_data(self):
        pass

    @staticmethod
    def get_event_info(event_data):
        """ Gets markers and timestamps from event data
        Ex:
        event_data = OpenEphys.load('all_channels.events')
        markers, timestamps = get_event_info(event_data)

        Required parameters:
        event_data -- valid open-ephys dict containing event data, obtained via OpenEphys.load (dict)

        Returns:
        markers -- list of markers (list of float)
        timestamps -- list of timestamps (list of float)
        """
        markers = []
        timestamps = []
        marker_val = 0
        for timestamp, channel, eventId in zip(event_data['timestamps'], 2**(7 - event_data['channel']), event_data['eventId']):
            if eventId == 1:
                marker_val += channel
            else:
                if marker_val:
                    markers.append(marker_val)
                    timestamps.append(timestamp)
                marker_val = 0
        return markers, timestamps
    
    def to_nex(self, nexfile_path, timestampFrequency = None, **params):
        nexfile_path = Path(nexfile_path)
        assert nexfile_path.suffix == '.nex', 'File must have extension ".nex"'

        if timestampFrequency is None:
            #infer timestampFrequency from recipe 
            pass
        nexfile = Nex(nexfile_path, **params)
        
        nexfile.open('w', timestampFrequency=timestampFrequency)
        start_time = time.time()

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
    #         wire = i,
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