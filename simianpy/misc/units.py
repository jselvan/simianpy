def get_scale_factor(input_units, sampling_rate=None):
    if input_units=='ms':
        return 1e3
    elif input_units=='s':
        return 1
    elif input_units=='samples':
        if sampling_rate is None:
            raise ValueError('Must provide sampling_rate if input_units=="samples"')
        return sampling_rate