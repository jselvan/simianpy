import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_trial_contdata(trial_info, contdata, tracelength=None, start='trialstart', 
    end='trialend', pad=(None, None), cols=None, relative_timestamps=True, simple_index=False,
    dropna=False):
    """Get eye traces for each trial

    Parameters
    ----------
    trial_info: pd.DataFrame
        output of BehaviouralData.get_trial_info()
    contdata: pd.DataFrame
        continuous data to extract into trial structure
    tracelength: pd.Timedelta or None, default: None
        Determines length of eye traces
        If not None, provided 'end' is ignored
    start: str, default: 'trialstart'
        A string that corresponds to a marker name used as the start of the slice
    end: str, default: 'trialend'
        A string that corresponds to a marker name used as the end of the slice
        Ignored if tracelength is provided
    pad: 2-tuple of pd.Timedelta or None, default: (None, None)
        A 2-tuple corresponding to padding at start and end of slice
    cols: list or None, default: None
        List of columns to extract from self.contdata
    relative_timestamps: boolean, default: True
        if true, timestamps are subtracted by start value
    dropna: boolean, default: False
        if true, any trials with timestamps in start or end that are nan values will be dropped 
    
    Returns
    -------
    trial_contdata: pd.DataFrame
    """
    trial_info['slice_start'] = trial_info[start]
    trial_info['slice_end'] = trial_info[end]

    if dropna:
        idx = trial_info.loc[:, ['slice_start', 'slice_end']].isna().any(axis=1)
        trial_info = trial_info[~idx]
    
    if cols is None:
        cols = slice(None)

    def get_contdata(row):
        start = row.slice_start
        if pad[0] is not None:
            start += pad[0]
        end = row.slice_end if tracelength is None else row.slice_start+tracelength
        if pad[1] is not None:
            end += pad[1]
        tslice = slice(start, end)

        trace = contdata.loc[tslice, cols]
        if relative_timestamps:
            trace.index = trace.index - row.slice_start
        
        return trace

    if simple_index:
        trial_contdata = trial_info.apply(get_contdata,axis=1)
        trial_contdata_df = pd.concat(trial_contdata.to_dict(), names=['trialid',*contdata.index.names])
    else:
        trial_contdata = {
            (row.Index, row.condition, row.outcome): get_contdata(row)
            for row in trial_info.itertuples()
        }

        names = ('trialid', 'condition', 'outcome', *contdata.index.names)
        trial_contdata_df = pd.concat(trial_contdata, names=names)
    return trial_contdata_df

class BehaviouralData:
    """Utility class to analyze Behavioural Data

    Takes event data and (optionally) continuous data to get trial by trial \
    information based on a config
    Trial start and end codes are used to extract trials
        value for the start code becomes the condition name
        value for the end code becomes the outcome variable
        timestamp is collected for all codes that fall within a trial

    Parameters
    ----------
    event_data: pd.Series
        event data with a datetime index and values corresponding to event markers
    config: dict
        config dict with a map of all codes as well as start and end events
    contdata: pd.DataFrame
        continuous data with a datetime index
        may contain eye data, LFP data, etc
    
    Attributes
    ----------
    trialstart
    trialend

    Methods
    -------
    get_trial_info()
    get_trial_contdata(tracelength=None)

    Examples
    --------
    config = {
        'start': [1,2], 
        'end': [200,201], 
        'codes': {
            1:'left', 2:'right', 200:'correct', 
            201:'incorrect', 50: 'stimulus onset'
        } 
    }
    --coming soon--
    """
    def __init__(self, event_data, config, contdata=None):
        self.event_data = event_data
        self.contdata = contdata

        params = ('codes', 'start', 'end')
        if not all(param in config.keys() for param in params):
            raise ValueError(f"Config must be a dict-like with keys: {params}")
        unique_markers_in_file = set(self.event_data.unique())
        unique_markers_in_config = set(config['codes'].keys())
        if not all(unique_marker in config['codes'] for unique_marker in unique_markers_in_file):
            missing_markers = unique_markers_in_file - unique_markers_in_config
            warnings.warn(f"""All markers should be described in configs.
            Here are the markers in the file {self.event_data.unique()}.
            The following keys are missing from the config {missing_markers}""")
        self.config = config
    
    @property
    def trialstart(self):
        return self.event_data[self.event_data.isin(self.config['start'])].index

    @property
    def trialend(self):
        trialstartidx, = np.where(self.event_data.isin(self.config['start']))
        return self.event_data.index[(trialstartidx[1:]-1)].append(self.event_data.index[-1:])

    def get_trial_info(self):
        trials = [self.event_data.loc[slice(start,end)] for start, end in zip (self.trialstart, self.trialend)]
        trialinfo = []
        for trial in trials:
            _trialinfo = {}
            for timestamp, marker in trial.items():
                marker_value = self.config['codes'].get(marker, marker)
                if marker in self.config['start']:
                    _trialinfo['condition'] = marker_value
                    _trialinfo['trialstart'] = timestamp 
                elif marker in self.config['end']:
                    _trialinfo['outcome'] = marker_value
                    _trialinfo['trialend'] = timestamp
                else:
                    _trialinfo[marker_value] = timestamp
            trialinfo.append(_trialinfo)
        trialinfo_df = pd.DataFrame(trialinfo)
        trialinfo_df.index.name = 'trialid'
        return trialinfo_df
    
    def get_trial_contdata(self, **kwargs):
        if self.contdata is None:
            raise ValueError('self.contdata must be provided to use this function')
        return get_trial_contdata(kwargs.get('trial_info', self.get_trial_info()), self.contdata)

#TODO: implement getting spike_data from behaviouraldata? Or should I make a parent class for this, and subclass behavioural data separately from one that can do more?
# def get_spikes_by_event(event_timestamps, spike_data, pad=(None,None)):
#     left, right = pad
#     def get_spike(timestamp):
#         spike = spike_data.loc[slice(timestamp+left,timestamp+right)]
#         spike_timestamps = (spike.index - timestamp).total_seconds() * 1e3
#         return spike_timestamps
#     return {event: get_spike(timestamp) for event, timestamp in event_timestamps.items()}
        
