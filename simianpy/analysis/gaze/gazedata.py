from itertools import product
from typing import Callable, Dict, Literal, Mapping, Optional, Sequence
from typing import SupportsFloat as Number

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from simianpy.misc import binary_digitize

query_fmt = Mapping[Literal["min", "max"], Number]


def parse_query(x: ArrayLike, query: Optional[query_fmt] = None):
    x = np.asarray(x)
    if query == None:
        query = {}
    min_ = query.get("min", -np.inf)
    max_ = query.get("max", np.inf)
    return (min_ < x) & (x < max_)


class GazeData:
    def __init__(self, data: xr.DataArray, blink_mask: Optional[np.ndarray] = None):
        self.data = data
        if blink_mask is None:
            blink_mask = np.ones(self.data.time.size, dtype=bool)
        self.blink_mask = blink_mask
        self.inferred: Dict[str, pd.DataFrame] = {}

    @classmethod
    def from_arrays(
        cls,
        time: np.ndarray,
        position: np.ndarray,
        dimensions: Sequence[str],
        trialid: Optional[np.ndarray] = None,
        blink_mask: Optional[np.ndarray] = None,
    ):
        data = xr.DataArray(
            position,
            dims=("time", "dimension"),
            coords=dict(time=time, dimension=dimensions),
        )
        if trialid is not None:
            data = data.assign_coords(trialid=("time", trialid))
        return cls(data, blink_mask)

    def mask_blinks(self, threshold: Number = 30, pad: Optional[int] = None):
        """Mask blinks in gaze data

        More specifically, this mask will hide all data points that are "off the screen" as defined by the threshold.

        Parameters
        ----------
        threshold : Number, optional
            The absolute value threshold for gaze events off the screen, by default 30
        pad : int, optional
            The number of samples around "blink" events to mask, by default None
        """
        blink_start, blink_end = binary_digitize(
            (np.abs(self.data) >= threshold).any("dimension")
        )
        if pad is not None:
            blink_start = np.clip(
                blink_start - pad, 0, self.blink_mask.size, out=blink_start
            )
            blink_end = np.clip(blink_end + pad, 0, self.blink_mask.size, out=blink_end)
        if blink_start.size != 0:
            idx = np.concatenate(
                [np.arange(start, end) for start, end in zip(blink_start, blink_end)]
            )
        self.blink_mask[idx] = False

    def differentiate(
        self,
        diff_method: Literal["radial"] = "radial",
        diff_dimensions: Optional[str | Sequence[str]] = None,
        pre_filter_method: Optional[Callable[[xr.DataArray], xr.DataArray]] = None,
        post_filter_method: Optional[Callable[[xr.DataArray], xr.DataArray]] = None,
    ) -> np.ndarray:
        """Differentiate gaze data

        Parameters
        ----------
        diff_method : str, optional
            radial distance is the only method supported
            by default "radial"
        diff_dimensions : str | list[str] | None, optional
            Select a single or a subset of dimensions
            If None, selects all dimensions, by default None
        (pre_/post_)filter_method : Callable, optional
            If provided, applies this function to the data before or after differentiation
            by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        time = np.diff(self.data.time)
        if diff_method == "radial":
            if diff_dimensions is None:
                data = self.data
            else:
                data = self.data.sel(dimension=diff_dimensions)
            if pre_filter_method is not None:
                data = xr.apply_ufunc(
                    pre_filter_method,
                    data,
                    input_core_dims=[["dimension", "time"]],
                    output_core_dims=[["dimension", "time"]],
                )
            difference = data.diff("time")
            if post_filter_method is not None:
                difference = xr.apply_ufunc(
                    post_filter_method,
                    difference,
                    input_core_dims=[["dimension", "time"]],
                    output_core_dims=[["dimension", "time"]],
                )
            difference = np.hypot(*difference.transpose("dimension", "time"))
        else:
            raise ValueError
        diff = difference / time
        return diff

    def identify_velocity_events(self, velocity_query, velocity_params={}):
        velocity = self.differentiate(**velocity_params)
        velocity = np.abs(velocity)
        mask = parse_query(velocity, velocity_query)
        onsets, offsets = binary_digitize(mask)
        return {"onset": onsets, "offset": offsets, "velocity": velocity}

    def get_saccades(
        self,
        velocity_query: query_fmt,
        duration_query: Optional[query_fmt] = None,
        peak_velocity_query: Optional[query_fmt] = None,
        amplitude_query: Optional[query_fmt] = None,
        velocity_params={},
    ):
        data = self.identify_velocity_events(velocity_query, velocity_params)
        dimensions = [dim.item() for dim in self.data.dimension]

        # compute some derived quantities
        computed = {}
        for field in ["onset", "offset"]:
            computed[f"{field}_time"] = self.data.time[data[field]].values
            if 'trialid' in self.data.coords:
                computed[f"{field}_trialid"] = self.data.trialid[data[field]].values
            for dim in dimensions:
                computed[f"{field}_{dim}"] = (
                    self.data.isel(time=data[field]).sel(dimension=dim).values
                )
        computed["duration"] = computed["offset_time"] - computed["onset_time"]
        computed["amplitude"] = np.hypot(
            *[
                computed[f"offset_{dim}"] - computed[f"onset_{dim}"]
                for dim in dimensions
            ]
        )

        # indices = np.stack([data['onset'], data['offset']]).T.flatten()
        indices = np.concatenate(
            [
                np.arange(onset, offset)
                for (onset, offset) in zip(data["onset"], data["offset"])
            ]
        )
        saccade_id = np.concatenate(
            [
                np.ones(offset - onset) * idx
                for idx, (onset, offset) in enumerate(
                    zip(data["onset"], data["offset"])
                )
            ]
        )
        # computed['trace_ids'] = saccade_id
        position_traces = self.data.isel(time=indices).values
        velocity_traces = data["velocity"][indices]
        # support exporting the traces
        lengths = data["offset"] - data["onset"]
        # velocity_traces_split = np.split(velocity_traces, np.cumsum(lengths[:-1]))
        order = np.lexsort((velocity_traces, saccade_id))
        index = np.zeros(saccade_id.size, dtype=bool)
        index[-1] = True
        index[:-1] = saccade_id[1:] != saccade_id[:-1]
        computed["peak_velocity"] = velocity_traces[order][index]

        # if queries are provided, filter the data
        query_mask = (
            parse_query(computed["duration"], duration_query)
            & parse_query(computed["peak_velocity"], peak_velocity_query)
            & parse_query(computed["amplitude"], amplitude_query)
        )
        query_idx = np.where(query_mask)[0]
        for key, value in computed.items():
            computed[key] = value[query_idx]

        computed = pd.DataFrame(computed)
        self.inferred["saccades"] = computed

        return computed

    def get_fixations(
        self,
        velocity_query: query_fmt,
        duration_query: Optional[query_fmt] = None,
        velocity_params={},
    ):
        data = self.identify_velocity_events(velocity_query, velocity_params)
        dimensions = [dim.item() for dim in self.data.dimension]

        # compute some derived quantities
        computed = {}
        for field in ["onset", "offset"]:
            computed[f"{field}_time"] = self.data.time[data[field]].values
        computed["duration"] = computed["offset_time"] - computed["onset_time"]

        # TODO: maybe mask fixations before computing the positions

        # compute the mean position in case of drift
        indices = np.concatenate(
            [
                np.arange(onset, offset)
                for (onset, offset) in zip(data["onset"], data["offset"])
            ]
        )
        fixation_id = np.concatenate(
            [
                np.ones(offset - onset) * idx
                for idx, (onset, offset) in enumerate(
                    zip(data["onset"], data["offset"])
                )
            ]
        )
        position_traces = self.data.isel(time=indices).values

        counts = data["offset"] - data["onset"]
        indices = np.insert(np.cumsum(counts[:-1]), 0, 0)
        mean_position = (
            np.add.reduceat(position_traces, indices, axis=0) / counts[:, None]
        )
        for i, dim in enumerate(dimensions):
            computed[dim] = mean_position[:, i]

        query_mask = parse_query(computed["duration"], duration_query)
        query_idx = np.where(query_mask)[0]
        for key, value in computed.items():
            computed[key] = value[query_idx]

        computed = pd.DataFrame(computed)

        self.inferred["fixations"] = computed

        return computed

    def view(self):
        try:
            from pyqtmgl.test.runner import run_dockable
            from pyqtmgl.widgets.continuous_viewer import ContinuousViewer
        except ImportError:
            raise ImportError(
                "pyqtmgl is not installed. Please install it to use this function."
            )

        ORANGE = (1, 0.5, 0)
        BLUE = (0, 0, 1)

        WHITE = (1, 1, 1)
        RED = (1, 0, 0)
        SACCADE_COLOUR = WHITE
        FIX_COLOUR = RED
        points = self.data.values.T
        colours = np.ones([points.shape[0], points.shape[1], 3])
        colours[0, :] = ORANGE
        colours[1, :] = BLUE
        for key, data in self.inferred.items():
            for onset_t, offset_t in zip(data["onset_time"], data["offset_time"]):
                onset = self.data.get_index("time").get_loc(onset_t)
                offset = self.data.get_index("time").get_loc(offset_t)
                tslice = slice(onset, offset)
                if key == "saccades":
                    colours[:, tslice] = SACCADE_COLOUR
                elif key == "fixations":
                    colours[:, tslice] = FIX_COLOUR
                else:
                    raise ValueError(f"Unknown event type: {key}")

        run_dockable([ContinuousViewer], points=points, colours=colours)

    def save(self, path: str):
        """Save the gaze data to a zip file

        Parameters
        ----------
        path : str
            The path to save the data to
        """
        import json
        import os
        import zipfile

        with zipfile.ZipFile(path, "w") as zf:
            # Save the data
            with zf.open("time.npy", "w") as f:
                np.save(f, self.data.time)
            with zf.open("position.npy", "w") as f:
                np.save(f, self.data.values)
            zf.writestr(
                "dimensions.txt", "\n".join(self.data.dimension.values).encode("utf-8")
            )
            # Save the blink mask
            np.save(zf.open("blink_mask.npy", "w"), self.blink_mask)
            # Save the inferred events
            for key, value in self.inferred.items():
                with zf.open(f"{key}.csv", "w") as f:
                    value.to_csv(f, index=False)

    @classmethod
    def load(cls, path: str):
        """Load the gaze data from a zip file

        Parameters
        ----------
        path : str
            The path to load the data from

        Returns
        -------
        GazeData
            The loaded gaze data
        """
        import json
        import os
        import zipfile

        with zipfile.ZipFile(path, "r") as zf:
            # Load the data
            time = np.load(zf.open("time.npy"))
            position = np.load(zf.open("position.npy"))
            with zf.open("dimensions.txt") as f:
                dimensions = f.read().decode("utf-8").split("\n")

            # Load the blink mask
            blink_mask = np.load(zf.open("blink_mask.npy"))
            # Load the inferred events
            inferred = {}
            for name in zf.namelist():
                if name.endswith(".csv"):
                    event = name.replace(".csv", "")
                    with zf.open(name) as f:
                        inferred[event] = pd.read_csv(f)

        gd = cls.from_arrays(time, position, dimensions, blink_mask)
        gd.inferred = inferred
        return gd

    @classmethod
    def from_monkeylogic(cls, path: str, align_event: Optional[int]=None):
        """Load gaze data from a MonkeyLogic .gaze file

        Parameters
        ----------
        path : str
            The path to the .gaze file

        Returns
        -------
        GazeData
            The loaded gaze data
        """
        from pathlib import Path

        from simianpy.io.monkeylogic.mlread import load

        data = load(Path(path), include_user_vars=False)

        if align_event is None:
            eyedata = np.concatenate(
                [
                    np.c_[
                        np.repeat(trial["trialid"], trial["eye"].shape[-1]),
                        np.arange(trial["eye"].shape[-1]) + trial["start_time"],
                        trial["eye"].T,
                    ]
                    for trial in data
                ]
            )
        else:
            eyedata = np.concatenate(
                [
                    np.c_[
                        np.repeat(trial["trialid"], trial["eye"].shape[-1]),
                        np.arange(trial["eye"].shape[-1]) - trial["timestamps"][trial["markers"]==align_event][0],
                        trial["eye"].T,
                    ]
                    for trial in data
                ]
            )
        trialid = eyedata[:, 0].astype('int16')
        time = eyedata[:, 1] / 1000  # convert to seconds
        eye = eyedata[:, 2:]
        gaze = GazeData.from_arrays(time, eye, ["eyeh", "eyev"], trialid=trialid)
        return gaze
