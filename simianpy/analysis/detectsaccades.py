import numpy as np
import pandas as pd

def DetectSaccades(eyedata, method = 'radial', velocity_threshold = 30, duration_threshold = None):
    """Detects saccades in eyedata using a velocity threshold (and optionally a duration threshold)

    Required arguments:
    eyedata (pandas.DataFrame) - must contain a pd.DatetimeIndex and columns 'eyeh' and 'eyev'

    Optional arguments:
    method (str; default = 'radial') - what is used to compute velocity; must be one of 'radial', 'horizontal', 'vertical'
    velocity_threshold (float; default = 30) - the velocity threshold in degrees per second
    duration_threshold (pd.Timedelta or None; default = None) - if not None, duration_threshold specifies minimum saccade duration
    """
    assert isinstance(eyedata, pd.DataFrame), f'eyedata must be pandas.DataFrame not {type(eyedata)}'
    assert isinstance(eyedata.index, pd.DatetimeIndex), f'eyedata.index must be pandas.DatetimeIndex not {type(eyedata.index)}'

    for col in ['eyeh', 'eyev']:
        assert col in eyedata.columns, f'Could not find {col} in eyedata'

    diff = eyedata.diff()

    if method == 'radial':
        velocity = np.hypot(diff['eyeh'], diff['eyev'])
    elif method == 'horizontal':
        velocity = diff['eyeh']
    elif method == 'vertical':
        velocity = diff['eyev']
    else:
        raise ValueError(f"Method must be one of ['radial', 'horizontal', 'vertical'] not {method}")

    velocity.rename('velocity')
    if velocity.index.freq is None:
        unique_deltas = pd.to_timedelta(np.diff(velocity.index)).unique()
        if unique_deltas.size == 1:
            freq, = unique_deltas
        else:
            raise ValueError('Provided eyedata does not have a continuous datetime index')
    else:
        freq = pd.to_timedelta(velocity.index.freq)
    saccade = velocity.abs() > ( velocity_threshold * freq.total_seconds() )

    onset = saccade & (saccade != saccade.shift()) & ~saccade.shift().isna()
    offset = ~saccade & (saccade != saccade.shift()) & ~saccade.shift().isna()

    onset = saccade[onset].index
    offset = saccade[offset].index

    if len(onset) != len(offset):
        onset, offset = zip(*zip(onset, offset))

    saccade_data = pd.DataFrame({
        'onset': onset,
        'offset': offset
    })

    if duration_threshold is None:
        return saccade_data
    else:
        return saccade_data[(saccade_data['offset'] - saccade_data['onset']) > duration_threshold]