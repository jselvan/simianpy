from typing import Dict, Mapping, Optional, Sequence, SupportsFloat

import numpy as np
import pandas as pd
import xarray as xr

from simianpy.analysis.gaze.gazedata import GazeData, parse_query

query_fmt = Mapping[str, SupportsFloat]

import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


class TrialGazeData:
    def __init__(
        self,
        data: xr.DataArray,
        blink_mask: Optional[np.ndarray | xr.DataArray] = None,
        window: Optional[tuple[float, float]] = None,
        trial_metadata: Optional[pd.DataFrame] = None,
        inferred: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        if tuple(data.dims) != ("trialid", "time", "dimension"):
            raise ValueError("TrialGazeData data must have dims ('trialid', 'time', 'dimension')")
        self.data = data
        if blink_mask is None:
            blink_mask = np.ones((data.sizes["trialid"], data.sizes["time"]), dtype=bool)
        if isinstance(blink_mask, xr.DataArray):
            blink_mask = blink_mask.transpose("trialid", "time").values
        blink_mask = np.asarray(blink_mask, dtype=bool)
        if blink_mask.shape != (data.sizes["trialid"], data.sizes["time"]):
            raise ValueError("blink_mask must have shape (n_trials, n_time)")
        self.blink_mask = xr.DataArray(
            blink_mask,
            dims=("trialid", "time"),
            coords={"trialid": data.coords["trialid"], "time": data.coords["time"]},
        )
        if window is None:
            window = (float(data.time.min()), float(data.time.max()))
        self.window = window
        self.trial_metadata = None
        if trial_metadata is not None:
            self.set_trial_metadata(trial_metadata)
        self.inferred: Dict[str, pd.DataFrame] = inferred.copy() if inferred is not None else {}

    def __repr__(self):
        return (
            f"TrialGazeData(n_trials={self.data.sizes['trialid']}, "
            f"n_time={self.data.sizes['time']}, "
            f"n_dimensions={self.data.sizes['dimension']}, window={self.window})"
        )

    @staticmethod
    def _prepare_metadata(metadata: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)
        if metadata.index.has_duplicates:
            raise ValueError("Trial metadata index must be unique")
        unique_ids = pd.Index(np.unique(ids), name=metadata.index.name or "trialid")
        if metadata.empty:
            metadata = metadata.copy()
            metadata.index = unique_ids[:0]
            return metadata.reindex(unique_ids)
        if metadata.index.equals(pd.RangeIndex(start=0, stop=len(unique_ids), step=1)) and len(metadata) == len(unique_ids):
            metadata = metadata.copy()
            metadata.index = unique_ids
            metadata.index.name = "trialid"
            return metadata
        return metadata.reindex(unique_ids)

    def set_trial_metadata(self, trial_metadata: pd.DataFrame) -> pd.DataFrame:
        trial_metadata = self._prepare_metadata(trial_metadata, self.data.trialid.values)
        self.trial_metadata = trial_metadata
        self.data = self.data.assign_coords(
            {
                col: ("trialid", trial_metadata.loc[self.data.trialid.values, col].values)
                for col in trial_metadata.columns
            }
        )
        self.blink_mask = self.blink_mask.assign_coords(
            {
                col: ("trialid", trial_metadata.loc[self.data.trialid.values, col].values)
                for col in trial_metadata.columns
            }
        )
        return self.trial_metadata

    def add_trial_metadata(self, trial_metadata: pd.DataFrame) -> pd.DataFrame:
        trial_metadata = self._prepare_metadata(trial_metadata, self.data.trialid.values)
        if self.trial_metadata is None:
            return self.set_trial_metadata(trial_metadata)
        self.trial_metadata = self.trial_metadata.copy()
        for column in trial_metadata.columns:
            self.trial_metadata[column] = trial_metadata[column]
        return self.set_trial_metadata(self.trial_metadata)

    def _masked_trial(self, trialid):
        trial = self.data.sel(trialid=trialid).copy()
        valid = self.blink_mask.sel(trialid=trialid).values
        trial.values[~valid, :] = np.nan
        trial = trial.assign_coords(
            trialid=("time", np.repeat(trialid, trial.sizes["time"]))
        )
        return trial

    def _infer_events(self, event_type: str, *args, **kwargs) -> pd.DataFrame:
        frames = []
        dimensions = [dim.item() if hasattr(dim, "item") else dim for dim in self.data.dimension.values]
        trialids = self.data.trialid.values
        for trialid in trialids:
            trial = self._masked_trial(trialid)
            gd = GazeData(trial)
            if event_type == "saccades":
                frame = gd.get_saccades(*args, **kwargs)
            elif event_type == "fixations":
                frame = gd.get_fixations(*args, **kwargs)
            else:
                raise ValueError(f"Unknown event type: {event_type}")
            if frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "trialid", trialid)
            for dim in dimensions:
                if f"offset_{dim}" in frame.columns:
                    frame[f"landing_{dim}"] = frame[f"offset_{dim}"]
            frames.append(frame)
        output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"trialid": pd.Series(dtype=trialids.dtype)})
        self.inferred[event_type] = output
        return output

    def get_saccades(
        self,
        velocity_query: query_fmt,
        duration_query: Optional[query_fmt] = None,
        peak_velocity_query: Optional[query_fmt] = None,
        amplitude_query: Optional[query_fmt] = None,
        velocity_params={},
    ) -> pd.DataFrame:
        return self._infer_events(
            "saccades",
            velocity_query,
            duration_query=duration_query,
            peak_velocity_query=peak_velocity_query,
            amplitude_query=amplitude_query,
            velocity_params=velocity_params,
        )

    def get_fixations(
        self,
        velocity_query: query_fmt,
        duration_query: Optional[query_fmt] = None,
        velocity_params={},
    ) -> pd.DataFrame:
        return self._infer_events(
            "fixations",
            velocity_query,
            duration_query=duration_query,
            velocity_params=velocity_params,
        )

    def _join_trial_metadata(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.trial_metadata is None or events.empty:
            return events
        return events.join(self.trial_metadata, on="trialid")

    def _select_events(
        self,
        event_type: str,
        trialid: Optional[Sequence] = None,
        time_query: Optional[query_fmt] = None,
        time_field: Optional[str] = None,
        duration_query: Optional[query_fmt] = None,
        amplitude_query: Optional[query_fmt] = None,
        peak_velocity_query: Optional[query_fmt] = None,
        center: Optional[Sequence[float]] = None,
        distance: Optional[float] = None,
        anchor: Optional[str] = None,
        include_trial_metadata: bool = False,
    ) -> pd.DataFrame:
        if event_type not in self.inferred:
            raise ValueError(f"No inferred {event_type} found. Run get_{event_type} first.")
        events = self.inferred[event_type].copy()
        if events.empty:
            return self._join_trial_metadata(events) if include_trial_metadata else events

        mask = np.ones(len(events), dtype=bool)
        if trialid is not None:
            mask &= events["trialid"].isin(np.asarray(trialid))
        if duration_query is not None and "duration" in events:
            mask &= parse_query(events["duration"], duration_query)
        if amplitude_query is not None and "amplitude" in events:
            mask &= parse_query(events["amplitude"], amplitude_query)
        if peak_velocity_query is not None and "peak_velocity" in events:
            mask &= parse_query(events["peak_velocity"], peak_velocity_query)
        if time_query is not None:
            if time_field is None:
                time_field = "offset_time" if event_type == "saccades" else "onset_time"
            mask &= parse_query(events[time_field], time_query)
        if center is not None or distance is not None:
            if center is None or distance is None:
                raise ValueError("Both center and distance must be provided for spatial selection")
            dimensions = [dim.item() if hasattr(dim, "item") else dim for dim in self.data.dimension.values]
            if anchor is None:
                anchor = "landing" if event_type == "saccades" else ""
            if anchor:
                columns = [f"{anchor}_{dim}" for dim in dimensions]
            else:
                columns = dimensions
            coords = events[columns].to_numpy(dtype=float)
            mask &= np.linalg.norm(coords - np.asarray(center, dtype=float), axis=1) <= distance

        output = events.loc[mask].reset_index(drop=True)
        if include_trial_metadata:
            output = self._join_trial_metadata(output)
        return output

    def select_saccades(
        self,
        trialid: Optional[Sequence] = None,
        time_query: Optional[query_fmt] = None,
        time_field: str = "offset_time",
        duration_query: Optional[query_fmt] = None,
        amplitude_query: Optional[query_fmt] = None,
        peak_velocity_query: Optional[query_fmt] = None,
        center: Optional[Sequence[float]] = None,
        distance: Optional[float] = None,
        anchor: str = "landing",
        include_trial_metadata: bool = False,
    ) -> pd.DataFrame:
        return self._select_events(
            "saccades",
            trialid=trialid,
            time_query=time_query,
            time_field=time_field,
            duration_query=duration_query,
            amplitude_query=amplitude_query,
            peak_velocity_query=peak_velocity_query,
            center=center,
            distance=distance,
            anchor=anchor,
            include_trial_metadata=include_trial_metadata,
        )

    def select_fixations(
        self,
        trialid: Optional[Sequence] = None,
        time_query: Optional[query_fmt] = None,
        time_field: str = "onset_time",
        duration_query: Optional[query_fmt] = None,
        center: Optional[Sequence[float]] = None,
        distance: Optional[float] = None,
        anchor: str = "",
        include_trial_metadata: bool = False,
    ) -> pd.DataFrame:
        return self._select_events(
            "fixations",
            trialid=trialid,
            time_query=time_query,
            time_field=time_field,
            duration_query=duration_query,
            center=center,
            distance=distance,
            anchor=anchor,
            include_trial_metadata=include_trial_metadata,
        )

    def plot_time(
        self,
        trialid: Optional[Sequence] = None,
        dimensions: Optional[str | Sequence[str]] = None,
        ax=None,
        overlay_events: Optional[pd.DataFrame] = None,
        alpha: float = 0.7,
        mask_blinks: bool = False,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        data = self.data
        if trialid is not None:
            data = data.sel(trialid=trialid)
        if dimensions is not None:
            data = data.sel(dimension=dimensions)
        if "trialid" not in data.dims:
            data = data.expand_dims(trialid=[data.coords["trialid"].item()])
        if "dimension" not in data.dims:
            data = data.expand_dims(dimension=[data.coords["dimension"].item()])
        for current_trialid in data.trialid.values:
            trial = data.sel(trialid=current_trialid)
            blink_mask = self.blink_mask.sel(trialid=current_trialid).values if mask_blinks else None
            for dimension in trial.dimension.values:
                values = trial.sel(dimension=dimension).values.copy()
                if blink_mask is not None:
                    values[~blink_mask] = np.nan
                ax.plot(
                    trial.time.values,
                    values,
                    alpha=alpha,
                    label=f"{current_trialid}-{dimension}",
                )
        if overlay_events is not None and not overlay_events.empty:
            for _, event in overlay_events.iterrows():
                onset = event.get("onset_time", np.nan)
                offset = event.get("offset_time", np.nan)
                if pd.notna(onset):
                    ax.axvline(onset, color="k", linestyle="--", alpha=0.25)
                if pd.notna(offset):
                    ax.axvline(offset, color="k", linestyle=":", alpha=0.25)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")
        return ax

    def plot_xy(
        self,
        trialid: Optional[Sequence] = None,
        ax=None,
        overlay_events: Optional[pd.DataFrame] = None,
        dimensions: Optional[Sequence[str]] = None,
        alpha: float = 0.7,
        mask_blinks: bool = False,
        hue_time=False,
        timeslice=None
    ):
        import matplotlib.pyplot as plt
        data = self.data
        if mask_blinks:
            data = data.where(self.blink_mask)

        if ax is None:
            _, ax = plt.subplots()
        if dimensions is None:
            dimensions = list(self.data.dimension.values[:2])
        xdim, ydim = dimensions
        if trialid is not None:
            data = data.sel(trialid=trialid)
        if "trialid" not in data.dims:
            data = data.expand_dims(trialid=[data.coords["trialid"].item()])
        if timeslice is not None:
            data = data.sel(time=timeslice)
        for current_trialid in data.trialid.values:
            trial = data.sel(trialid=current_trialid)
            x = trial.sel(dimension=xdim).values.copy()
            y = trial.sel(dimension=ydim).values.copy()
            t = trial.time.values if hue_time else None
            colored_line(
                x,
                y,
                t,
                ax,
                alpha=alpha,
                label=str(current_trialid),
            )
        if overlay_events is not None and not overlay_events.empty:
            xcol = f"landing_{xdim}" if f"landing_{xdim}" in overlay_events.columns else xdim
            ycol = f"landing_{ydim}" if f"landing_{ydim}" in overlay_events.columns else ydim
            ax.scatter(overlay_events[xcol], overlay_events[ycol], color="k", s=20)
        ax.set_xlabel(str(xdim))
        ax.set_ylabel(str(ydim))
        return ax
