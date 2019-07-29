import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BehaviouralData:
    """Utility class to analyze Behavioural Data

    Takes event data and eye data to get trial by trial information based on a config

    Parameters
    ----------
    event_data: pd.Series
        event data with a datetime index and values corresponding to event markers
    eyedata: pd.DataFrame
        eye data with a datetime index and containing 'eyeh' and 'eyev' columns
    config: dict
        config dict with a map of all codes as well as start and stop events
        ex: config = {'start': [1,2], 'end': [200,201], 'codes': {1:'left', 2:'right', 200:'correct', 201:'incorrect', 50: 'stimulus onset'} }
    
    Attributes
    ----------
    trialstart
    trialend
    trialinfo

    Methods
    -------
    get_trial_eyedata(tracelength=None)

    Examples
    --------
    --coming soon--
    """
    def __init__(self, event_data, eyedata, config):
        self.event_data = event_data
        self.eyedata = eyedata

        assert isinstance(config, dict)
        assert all(param in config for param in ['codes', 'start', 'end'])
        assert all(unique_marker in config['codes'] for unique_marker in self.event_data.unique()),  f"All markers must be described in configs. Here are the markers in the file {self.event_data.unique()}"
        self.config = config
    
    @property
    def trialstart(self):
        return self.event_data[self.event_data.isin(self.config['start'])].index

    @property
    def trialend(self):
        trialstartidx, = np.where(self.event_data.isin(self.config['start']))
        return self.event_data.index[(trialstartidx[1:]-1)].append(self.event_data.index[-1:])

    @property
    def trialinfo(self):
        trials = [self.event_data.loc[slice(start,end)] for start, end in zip (self.trialstart, self.trialend)]
        trialinfo = []
        for trial in trials:
            _trialinfo = {}
            for timestamp, marker in trial.items():
                if marker in self.config['start']:
                    _trialinfo['condition'] = self.config['codes'][marker]
                    _trialinfo['trialstart'] = timestamp 
                elif marker in self.config['end']:
                    _trialinfo['outcome'] = self.config['codes'][marker]
                    _trialinfo['trialend'] = timestamp
                else:
                    _trialinfo[self.config['codes'][marker]] = timestamp
            trialinfo.append(_trialinfo)
        return pd.DataFrame(trialinfo)
    
    def get_trial_eyedata(self, tracelength = None):
        """Get eye traces for each trial

        Parameters
        ----------
        tracelength: pd.Timedelta or None, default: None
            Determines length of eye traces, If None, get eye traces from trial start to trial end
        
        Returns
        -------
        trial_eyedata: list of pd.DataFrame
        """
        if tracelength is None:
            trial_eyedata = [self.eyedata.loc[slice(start, end)] for start, end in zip(self.trialstart, self.trialend)]
        else:
            trial_eyedata = [self.eyedata.loc[slice(start, start + tracelength)] for start in self.trialstart]
        
        return trial_eyedata