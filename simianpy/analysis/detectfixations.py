import numpy as np
import pandas as pd

def DetectFixations(eye_data, velocity_threshold=2, duration_threshold=None, sampling_rate=1e3, Filter=None):
    if Filter is None:
        velocity = eye_data.diff().abs()*sampling_rate
    else:
        velocity = (eye_data.apply(Filter).diff().abs()*sampling_rate).apply(Filter)
    fix = (velocity < velocity_threshold).all(axis=1)
    onset = np.where(~fix[:-1].values & fix[1:].values)
    offset = np.where(fix[:-1].values & ~fix[1:].values)

    onset = fix.index[onset]
    offset = fix.index[offset]
    #TODO: write a general function that accomplishes this (can be used for analog TTL, etc) 
    # onset, offset = binary_digitize(velocity.abs(), velocity_threshold, 'lt')
    if onset[0] > offset[0]:
        offset = offset[1:]
    if onset[-1] > offset[-1]:
        onset = onset[:-1]
        
    if onset.size != offset.size:
        raise ValueError(
            f"Number of onsets {onset.size} and offsets {offset.size} \
            must be the same or differ by 1 (edge case where the last \
            offset does not occur within the trace)")

    fixation_data = pd.DataFrame({
        'onset': onset,
        'offset': offset
    })
    
    fixation_data['duration_dt'] = fixation_data['offset'] - fixation_data['onset']
    
    if duration_threshold is not None:
        fixation_data = fixation_data[fixation_data['duration_dt'] > duration_threshold]
    
    if hasattr(fixation_data['duration_dt'], 'dt') and hasattr(fixation_data['onset_t'].dt, 'total_seconds'):
        fixation_data['duration'] = fixation_data['duration_dt'].dt.total_seconds() * 1e3

    fixation_data = fixation_data.join(
        fixation_data.apply(
            lambda fixation: eye_data.loc[slice(fixation.onset, fixation.offset),['eyeh','eyev']].mean(), 
            axis=1
        )
    )
    
    return fixation_data