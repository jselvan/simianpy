import pandas as pd

from ..misc import binary_digitize


def DetectFixations(
    eye_data,
    velocity_threshold=2,
    duration_threshold=None,
    sampling_rate=1e3,
    Filter=None,
):
    if Filter is None:
        velocity = eye_data.diff().abs() * sampling_rate
    else:
        velocity = (eye_data.apply(Filter).diff().abs() * sampling_rate).apply(Filter)
    fix = (velocity < velocity_threshold).all(axis=1)
    onset, offset = binary_digitize(fix)
    if onset.size == 0 or offset.size == 0:
        return pd.DataFrame()

    onset, offset = fix.index[onset], fix.index[offset]

    fixation_data = pd.DataFrame({"onset": onset, "offset": offset})

    fixation_data["duration_dt"] = fixation_data["offset"] - fixation_data["onset"]

    if duration_threshold is not None:
        fixation_data = fixation_data[fixation_data["duration_dt"] > duration_threshold]

    if hasattr(fixation_data["duration_dt"], "dt") and hasattr(
        fixation_data["duration_dt"].dt, "total_seconds"
    ):
        fixation_data["duration"] = (
            fixation_data["duration_dt"].dt.total_seconds() * 1e3
        )

    if fixation_data.empty:
        return fixation_data

    fixation_data = fixation_data.join(
        fixation_data.apply(
            lambda fixation: eye_data.loc[
                slice(fixation.onset, fixation.offset), ["eyeh", "eyev"]
            ].mean(),
            axis=1,
        )
    )
    fixation_data.index.name = "fixationid"
    return fixation_data
