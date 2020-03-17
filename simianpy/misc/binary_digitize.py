import numpy as np

def binary_digitize(data, threshold=None, errors=True):
    """ Infer on and off states from a continuous signal and determine transition points

    Parameters
    ----------
    data: array-like
        data is the continuous array-like to be digitized
        array of bool or array containing items that can be compared to threshold
    threshold: any comparable dtype, default: None
        if data is already a boolean array, do not supply a threshold
    errors: boolean, default: True
        If True:
        Will silently drop the first offset if not preceded by an onset 
        and the last onset if not followed by an offset

        If onsets.size != offsets.size, a ValueError is raised

    Notes
    -----
    onset is defined as the first sample being above threshold, or being true
    offset is defined as the first sample being equal to or below threshold, or being false

    Example
    -------
    # to classify TTL pulses from an analog input
    TTLdata = [0,0,0.1,3.3,4.6,4.7,4,8,3.2,0.1,0,0,0]
    onset, offset = binary_digitize(TTLdata, threshold=4)
    fig, ax = plt.subplots()
    ax.plot(TTLdata)
    for on, off in zip(onset, offset):
        pass

    """
    data = np.asarray(data)
    if threshold is None:
        on, off = data, ~data
    else:
        off = data <= threshold
        on = data > threshold
    onsets, = np.where(off[:-1] & on[1:])
    offsets, = np.where(on[:-1] & off[1:])

    if errors:
        if onsets[0] > offsets[0]:
            offsets = offsets[1:]
        if onsets[-1] > offsets[-1]:
            onsets = onsets[:-1]
        
        if onsets.size != offsets.size:
            raise ValueError(
                f"Number of onsets {onsets.size} and offsets {offsets.size} \
                must be the same or differ by 1 due to edge cases"
            )
    
    onsets, offsets = onsets+1, offsets+1 # corrects for off-by-one error
    return onsets, offsets