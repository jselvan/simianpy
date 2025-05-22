from os import PathLike

import h5py
import numpy as np
from tqdm import tqdm

def read_h5(filename: PathLike, logger=None, pbar: bool=False, include_user_vars: bool=True):
    with h5py.File(filename, 'r') as f:
        n_trials = int(f['ML/TrialRecord/CurrentTrialNumber'][0,0]) #type: ignore
        for trial in tqdm(range(n_trials), desc="Loading trials", disable=not pbar):
            trial_data = f[f'ML/Trial{trial+1}']
            condition = trial_data['Condition'][0,0] #type: ignore
            eye = np.asarray(trial_data['AnalogData/Eye'][()]) #type: ignore
            markers = trial_data['BehavioralCodes/CodeNumbers'][0, :] #type: ignore
            timestamps = trial_data['BehavioralCodes/CodeTimes'][0, :] #type: ignore
            trial_error = trial_data['TrialError'][0,0] #type: ignore
            start_time = float(trial_data['AbsoluteTrialStartTime'][0,0]) #type: ignore
            trial_info = {
                'trialid': trial+1,
                'condition': condition,
                'start_time': start_time,
                'eye': eye,
                'time': start_time + np.arange(eye.shape[1]),
                'markers': markers,
                'timestamps': timestamps,
                'trial_error': trial_error,
            }
            if include_user_vars:
                trial_info['user_vars'] = {
                    key: trial_data['UserVars'][key][()].squeeze() #type: ignore
                    for key in trial_data['UserVars'] #type: ignore
                    if key not in ['SkippedFrameTimeInfo']
                }
            yield trial_info