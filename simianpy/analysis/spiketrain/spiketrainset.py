from typing import Optional, Sequence, Tuple, Mapping
import zipfile
import io

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr


try:
    from simianpy.analysis.spiketrain.spiketrainsetviewer import launch_viewer
except ImportError:
    pyqtmgl = False
else:
    pyqtmgl = True

class SpikeTrainSet:
    def __init__(
        self,
        trialids: np.ndarray,
        unitids: np.ndarray,
        spike_times: np.ndarray,
        window: Optional[Tuple[float, float]] = None,
        trial_metadata: Optional[pd.DataFrame] = None,
        epochids: Optional[np.ndarray] = None,
        epoch_names: Optional[Sequence[str]] = None,
    ):
        """Initialize SpikeTrainSet

        Parameters
        ----------
        trialids : np.ndarray
            Array of trial ids
        unitids : np.ndarray
            Array of unit ids
        spike_times : np.ndarray
            Array of spike times
        window : tuple or None
            Tuple of (start, end) of window around event timestamps
            if None, use the range of spike_times

        Raises
        ------
        ValueError
            If trialids, unitids, and spike_times are not the same length
        """
        self.trialids = trialids
        self.n_trials = np.unique(trialids).size
        self.unitids = unitids
        self.n_units = np.unique(unitids).size
        self.spike_times = spike_times
        if window is None:
            window = (self.spike_times.min(), self.spike_times.max())
        self.window = window
        if not (len(trialids) == len(unitids) == len(spike_times)):
            raise ValueError("All inputs must be the same length")
        if trial_metadata is not None:
            trialids_unique = np.unique(trialids)
            trial_metadata = trial_metadata.loc[trialids_unique]
            self.trial_metadata = trial_metadata
        else:
            self.trial_metadata = None
        # Epoch support
        if epochids is not None:
            self.epochids = epochids
            self.n_epochs = np.unique(epochids).size
        else:
            self.epochids = None
            self.n_epochs = None
        self.epoch_names = epoch_names

    @classmethod
    def from_arrays(
        cls,
        spike_timestamps: ArrayLike,
        event_timestamps: ArrayLike,
        window: Tuple[float, float],
        spike_labels: Optional[ArrayLike] = None,
        event_labels: Optional[ArrayLike] = None,
        trial_metadata: Optional[pd.DataFrame] = None,
        epoch_names: Optional[Sequence[str]] = None,
    ) -> "SpikeTrainSet":
        """Generate SpikeTrainSet from arrays of spike and event timestamps

        Parameters
        ----------
        spike_timestamps : array_like
            Array of spike timestamps
        event_timestamps : array_like
            Array of event timestamps
        window : tuple
            Tuple of (start, end) of window around event timestamps
        spike_labels : array_like or None
            Array of spike labels
        event_labels : array_like or None
            Array of event labels
        epoch_names : list or None
            List of epoch names, must match number of epochs if provided

        Returns
        -------
        SpikeTrainSet
        """
        spike_timestamps = np.array(spike_timestamps)
        event_timestamps = np.array(event_timestamps)
        is_2d = event_timestamps.ndim == 2
        if spike_labels is None:
            spike_labels = np.zeros(len(spike_timestamps), dtype=np.int32)
        else:
            spike_labels = np.array(spike_labels)
        if event_labels is None:
            
            event_labels = np.arange(event_timestamps.shape[0], dtype=np.int32)
        else:
            event_labels = np.array(event_labels)
        if is_2d:
            n_events, n_epochs = event_timestamps.shape
            # Flatten event_timestamps, repeat event_labels and epochids
            event_timestamps_flat = event_timestamps.flatten()
            event_labels_flat = np.repeat(event_labels, n_epochs)
            epochids = np.tile(np.arange(n_epochs), n_events)
            if epoch_names is not None:
                epoch_names = list(epoch_names)
                if len(epoch_names) != n_epochs:
                    raise ValueError("Length of epoch_names must match number of epochs")
            else:
                epoch_names = [str(i) for i in range(n_epochs)]
        else:
            event_timestamps_flat = event_timestamps
            event_labels_flat = event_labels
            epochids = None
            epoch_names = None
        spk_sort_idx = np.argsort(spike_timestamps)
        evt_sort_idx = np.argsort(event_timestamps_flat)
        spike_timestamps = spike_timestamps[spk_sort_idx]
        spike_labels = spike_labels[spk_sort_idx]
        event_timestamps_flat = event_timestamps_flat[evt_sort_idx]
        event_labels_flat = event_labels_flat[evt_sort_idx]
        if epochids is not None:
            epochids = np.array(epochids)[evt_sort_idx]
        left, right = event_timestamps_flat + window[0], event_timestamps_flat + window[1]
        lidx = np.searchsorted(spike_timestamps, left, side="left")
        ridx = np.searchsorted(spike_timestamps, right, side="right")
        lengths = ridx - lidx
        offset = np.repeat(event_timestamps_flat, lengths)
        event_labels_rep = np.repeat(event_labels_flat, lengths)
        if epochids is not None:
            epochids_rep = np.repeat(epochids, lengths)
        else:
            epochids_rep = None
        spk_idx = np.concatenate([np.arange(l, r) for l, r in zip(lidx, ridx)])
        spike_times_aligned = spike_timestamps[spk_idx] - offset
        spike_labels_aligned = spike_labels[spk_idx]
        if epochids_rep is not None:
            return cls(event_labels_rep, spike_labels_aligned, spike_times_aligned, window, trial_metadata=trial_metadata, epochids=epochids_rep, epoch_names=epoch_names)
        else:
            return cls(event_labels_rep, spike_labels_aligned, spike_times_aligned, window, trial_metadata=trial_metadata)

    @classmethod
    def from_series(
        cls,
        spike_timestamps: pd.Series,
        event_timestamps: pd.Series,
        window: Tuple[float, float],
        trial_metadata: Optional[pd.DataFrame] = None,
    ):
        """Generate SpikeTrainSet from pd.Series of spike and event timestamps

        Parameters
        ----------
        spike_timestamps : pd.Series
            Series of spike timestamps with index of unit labels
        event_timestamps : pd.Series
            Series of event timestamps with index of event labels
        window : tuple
            Tuple of (start, end) of window around event timestamps

        Returns
        -------
        SpikeTrainSet
        """
        return cls.from_arrays(
            np.asarray(spike_timestamps.values),
            np.asarray(event_timestamps.values),
            window,
            spike_timestamps.index,
            event_timestamps.index,
            trial_metadata=trial_metadata,
        )

    def to_dataframe(self):
        """Convert spike train set to DataFrame

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns 'trialid', 'unitid', 'spike_time'
        """
        return pd.DataFrame(
            {
                "trialid": self.trialids,
                "unitid": self.unitids,
                "spike_time": self.spike_times,
            }
        )

    def to_psth(
        self, time_bins: Optional[Sequence] = None, time_step: Optional[float] = None
    ) -> xr.DataArray:
        """Convert spike train set to PSTH matrix
        
        Parameters
        ----------
        time_bins : sequence or None
            Sequence of time bins to use for the histogram. If None, will use time_step.
        time_step : float or None
            Time step to use for the histogram. If None, will calculate from time_bins.

        Raises
        ------
        ValueError
            If neither time_bins nor time_step is provided, or if they are not compatible

        Returns
        -------
        psth : xr.DataArray
            DataArray with dimensions ['trialid', 'unitid', 'epoch', 'time'] if epochids is provided,
            otherwise ['trialid', 'unitid', 'time']. Contains the counts of spikes in each bin.

        """
        if time_step is None:
            if time_bins is None:
                raise ValueError("Either time_bins or time_step must be provided")
            time_step = time_bins[1] - time_bins[0]
            time_bin_array = np.asarray(time_bins)
        else:
            left, right = self.window
            left = left - (time_step / 2)
            right = right + time_step
            time_bin_array = np.arange(left, right, time_step)
        trialids_unique, trialids_digitized = np.unique(
            self.trialids, return_inverse=True
        )
        unitids_unique, unitids_digitized = np.unique(self.unitids, return_inverse=True)
        time = time_bin_array[:-1] + (time_bin_array[1] - time_bin_array[0]) / 2
        coords = {
            "trialid": trialids_unique,
            "unitid": unitids_unique,
            "time": time,
        }
        if self.epochids is not None:
            epochids_unique, epochids_digitized = np.unique(
                self.epochids, return_inverse=True
            )
            # Create a 3D histogram for trialid, unitid, and epochid
            psth, _ = np.histogramdd(
                np.array([trialids_digitized, unitids_digitized, epochids_digitized, self.spike_times]).T,
                bins=[
                    np.arange(trialids_unique.size + 1),
                    np.arange(unitids_unique.size + 1),
                    np.arange(epochids_unique.size + 1),
                    time_bin_array,
                ]
            )
            coords["epoch"] = self.epoch_names if self.epoch_names is not None else epochids_unique
            dims = ["trialid", "unitid", "epoch", "time"]
        else:
            psth, _ = np.histogramdd(
                np.array([trialids_digitized, unitids_digitized, self.spike_times]).T,
                bins=[
                    np.arange(trialids_unique.size + 1),
                    np.arange(unitids_unique.size + 1),
                    time_bin_array,
                ]
            )
            dims = ["trialid", "unitid", "time"]
        psth = xr.DataArray(psth, coords=coords, dims=dims)
        # Attach trial metadata as coordinates if present
        if self.trial_metadata is not None:
            psth = psth.assign_coords({
                col: ("trialid", self.trial_metadata.loc[trialids_unique, col].values)
                for col in self.trial_metadata.columns
            })
        return psth

    def get_plotting_data(self, 
            group: Optional[str | Sequence[str]]=None, 
            sort: Optional[str | Sequence[str]]=None,
            palette: Optional[str | Mapping]=None,
            psth_params: Optional[Mapping]=None
        ):
        # if a sort parameter is provided, we will sort trials by those values
        # if grouped, we will sort by group at the end
        sort_keys = []
        output = {}
        unique_groups = np.ones(self.trialids.size, dtype=int)  # Default to single group if no grouping
        if self.trial_metadata is None:
            if sort is not None or group is not None:
                raise ValueError("Cannot sort or group trials without trial metadata")
            else:
                _, trialids = np.unique(self.trialids, return_inverse=True)
        else:
            if sort is not None:
                if isinstance(sort, str):
                    sort = [sort]
                if any(s not in self.trial_metadata.columns for s in sort):
                    raise ValueError(f"Sort columns {sort} not found in trial metadata")
                sort_keys.extend(sort)
            if group is not None:
                if isinstance(group, str):
                    group = [group]
                if any(g not in self.trial_metadata.columns for g in group):
                    raise ValueError(f"Group columns {group} not found in trial metadata")
                sort_keys.extend(group)

            if sort_keys:
                sorted_table = self.trial_metadata.sort_values(by=sort_keys, ascending=True)
                sorted_table['sorted_trialid'] = np.arange(len(sorted_table))
                trialids = sorted_table.loc[self.trialids, 'sorted_trialid'].values
            else:
                _, trialids = np.unique(self.trialids, return_inverse=True)
        
            if group is not None:
                groupdata = self.trial_metadata.loc[self.trialids, group].to_records(index=False) #type: ignore
                unique_groups, output['groupids'] = np.unique(groupdata, return_inverse=True)
                if palette is None:
                    palette = 'Set1'
                if isinstance(palette, str):
                    from matplotlib import colormaps
                    cmap = colormaps.get_cmap(palette)
                    if hasattr(cmap, 'colors'):
                        colors = cmap.colors #type: ignore
                    else:
                        colors = cmap(np.arange(cmap.N))

                    palette = {
                        tuple(g): colors[i % len(colors)]
                        for i, g in enumerate(unique_groups)
                    }
                colors = [palette.get(tuple(g), None) for g in unique_groups]  # ensure palette is an array
                output['colors'] = map(tuple, np.take(colors, output['groupids'], axis=0))

        if psth_params is None:
            psth_params = {}
        psth = self.to_psth(**psth_params)
        if group is not None:
            psth = psth.groupby(group).mean()
        else:
            psth = psth.mean(dim='trialid')

        output['trialid'] = trialids
        output['unitid'] = self.unitids
        output['spike_times'] = self.spike_times
        if self.epochids is not None:
            if self.epoch_names is not None:
                output['epoch'] = np.take(self.epoch_names, self.epochids)
            else:
                output['epoch'] = self.epochids

        if output.get('colors') is None:
            output['colors'] = [(0,0,0)] * len(output['trialid'])
            palette = {'default': (0, 0, 0)}
        output = pd.DataFrame(output)
        return output, psth, palette

    def get_firing_rates(
        self, window: Optional[Tuple[float, float]] = None
    ) -> xr.DataArray:
        if window is None:
            window = self.window
        # check that the window is within the range of the spike times
        if window[0] < self.window[0] or window[1] > self.window[1]:
            raise ValueError(
                "Window must be within the range of the provided SpikeTrainSet"
            )
        duration = window[1] - window[0]
        fr = self.to_psth(time_bins=window) / duration
        return fr

    def to_sdf(self, convolve, window="psp", window_size=None):
        pass

    def save(self, path: str):
        """Save the SpikeTrainSet as a DataArray in a zip file."""
        with zipfile.ZipFile(path, "w") as zf:
            with zf.open("spike_times.npy", "w") as f:
                np.save(f, self.spike_times)
            with zf.open("trialids.npy", "w") as f:
                np.save(f, self.trialids)
            with zf.open("unitids.npy", "w") as f:
                np.save(f, self.unitids)
            with zf.open("window.txt", "w") as f:
                value = f"{self.window[0]} {self.window[1]}"
                f.write(value.encode("utf-8"))
            # Save trial metadata as CSV if present
            if self.trial_metadata is not None:
                with zf.open("trial_metadata.csv", "w") as f:
                    csv_bytes = self.trial_metadata.to_csv().encode("utf-8")
                    f.write(csv_bytes)
            if self.epochids is not None:
                with zf.open("epochids.npy", "w") as f:
                    np.save(f, self.epochids)
            if self.epoch_names is not None:
                with zf.open("epoch_names.txt", "w") as f:
                    f.write("\n".join(self.epoch_names).encode("utf-8"))

    @classmethod
    def load(cls, path: str) -> "SpikeTrainSet":
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("spike_times.npy") as f:
                spike_times = np.load(f)
            with zf.open("trialids.npy") as f:
                trialids = np.load(f)
            with zf.open("unitids.npy") as f:
                unitids = np.load(f)
            with zf.open("window.txt") as f:
                window = tuple(map(float, f.read().decode("utf-8").split()))
                if len(window) != 2:
                    raise ValueError("Window must be a tuple of length 2")
            # Try to load trial metadata
            try:
                with zf.open("trial_metadata.csv") as f:
                    trial_metadata = pd.read_csv(io.StringIO(f.read().decode("utf-8")), index_col=0)
            except KeyError:
                trial_metadata = None
            
            # Try to load epochids
            try:
                with zf.open("epochids.npy") as f:
                    epochids = np.load(f)
            except KeyError:
                epochids = None
            # Try to load epoch names
            try:
                with zf.open("epoch_names.txt") as f:
                    epoch_names = f.read().decode("utf-8").splitlines()
            except KeyError:
                epoch_names = None
        return cls(trialids, unitids, spike_times, window, trial_metadata=trial_metadata, epochids=epochids, epoch_names=epoch_names)

    def view(self):
        if not pyqtmgl:
            raise ImportError(
                "pyqtmgl is not set up. Please install it to use this function."
                "You can install it with `pip install git+https://github.com/jselvan/pyqtmgl.git`."
            )
        launch_viewer(self)