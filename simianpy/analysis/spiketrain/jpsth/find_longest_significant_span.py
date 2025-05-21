import numpy as np

from simianpy.misc import binary_digitize


def find_longest_significant_span(covariogram, crit):
    lowest, highest = 0, covariogram.size

    # find onsets and offsets and handle errors manually
    onsets, offsets = binary_digitize(covariogram, crit, errors=False)

    if not (onsets.size or offsets.size):
        # if no onsets or offsets,
        # either all sig or none sig
        if (covariogram > crit).all():
            return [[lowest, highest]]
        else:
            return []
    else:
        if not onsets.size or offsets[0] < onsets[0]:
            onsets = np.append([lowest], onsets)
        if not offsets.size or offsets[-1] < onsets[-1]:
            offsets = np.append(offsets, [highest])

    if offsets.size != onsets.size:
        raise ValueError(
            f"size mismatch between {onsets=} and {offsets=} could not be fixed"
        )

    max_indices = np.argmax(offsets - onsets)
    return np.stack(onsets[max_indices], offsets[max_indices])
