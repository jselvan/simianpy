from functools import reduce
import pandas as pd


def parse_timeslice(timeslice, unit="ms"):
    for time in ("start", "end"):
        timepoint = timeslice.get(time)
        if timepoint is not None:
            if unit == "ms":
                timepoint = pd.to_timedelta(timepoint).total_seconds() * 1e3
            elif unit == "s":
                timepoint = pd.to_timedelta(timepoint).total_seconds()
        timeslice[time] = timepoint
    return slice(timeslice["start"], timeslice["end"])


class TimeSlice:
    def __init__(self, timeslice):
        if isinstance(timeslice, dict):
            timeslice = [timeslice]
        self.timeslice = []
        for t in timeslice:
            for time in ("start", "end"):
                timepoint = t.get(time)
                if timepoint is not None:
                    timepoint = pd.to_timedelta(timepoint).total_seconds()
                t[time] = timepoint
            self.timeslice.append(t)

    def as_slice(self, sampling_rate, as_int=False):
        timeslices = []
        for timeslice in self.timeslice:
            timeslice = timeslice.copy()
            if timeslice["start"] is not None:
                timeslice["start"] *= sampling_rate
                if as_int:
                    timeslice["start"] = int(timeslice["start"])
            if timeslice["end"] is not None:
                timeslice["end"] *= sampling_rate
                if as_int:
                    timeslice["end"] = int(timeslice["end"])
            timeslices.append(slice(timeslice["start"], timeslice["end"]))
        return timeslices

    def pandas_slice(self, pandas_obj, sampling_rate):
        timeslices = self.as_slice(sampling_rate)
        return pd.concat([pandas_obj.loc[t] for t in timeslices])

    def array_slice(self, array, sampling_rate):
        timeslices = self.as_slice(sampling_rate, as_int=True)
        return np.concatenate([array[t] for t in timeslices])

    def array_mask(self, array, sampling_rate):
        timeslices = self.as_slice(sampling_rate)
        mask = reduce(
            lambda x, y: x | y,
            (
                ((timeslice.start or array.min()) <= array)
                & (array <= (timeslice.stop or array.max()))
                for timeslice in timeslices
            ),
        )
        return array[mask]

    def __repr__(self):
        return str(self.timeslice)
