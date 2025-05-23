from typing import Optional, Sequence, Tuple
import zipfile

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr


class SpikeTrainSet:
    def __init__(
        self,
        trialids: np.ndarray,
        unitids: np.ndarray,
        spike_times: np.ndarray,
        window: Optional[Tuple[float, float]] = None,
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
        # ensure that they are all the same length
        if not (len(trialids) == len(unitids) == len(spike_times)):
            raise ValueError("All inputs must be the same length")

    @classmethod
    def from_arrays(
        cls,
        spike_timestamps: ArrayLike,
        event_timestamps: ArrayLike,
        window: Tuple[float, float],
        spike_labels: Optional[ArrayLike] = None,
        event_labels: Optional[ArrayLike] = None,
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

        Returns
        -------
        SpikeTrainSet
        """
        spike_timestamps = np.array(spike_timestamps)
        event_timestamps = np.array(event_timestamps)
        if spike_labels is None:
            spike_labels = np.zeros(len(spike_timestamps), dtype=np.int32)
        else:
            spike_labels = np.array(spike_labels)
        if event_labels is None:
            event_labels = np.arange(len(event_timestamps), dtype=np.int32)
        else:
            event_labels = np.array(event_labels)

        spk_sort_idx = np.argsort(spike_timestamps)
        evt_sort_idx = np.argsort(event_timestamps)
        spike_timestamps = spike_timestamps[spk_sort_idx]
        spike_labels = spike_labels[spk_sort_idx]
        event_timestamps = event_timestamps[evt_sort_idx]
        event_labels = event_labels[evt_sort_idx]

        left, right = event_timestamps + window[0], event_timestamps + window[1]
        lidx = np.searchsorted(spike_timestamps, left, side="left")
        ridx = np.searchsorted(spike_timestamps, right, side="right")
        lengths = ridx - lidx
        offset = np.repeat(event_timestamps, lengths)
        event_labels = np.repeat(event_labels, lengths)
        spk_idx = np.concatenate([np.arange(l, r) for l, r in zip(lidx, ridx)])
        spike_times_aligned = spike_timestamps[spk_idx] - offset
        spike_labels_aligned = spike_labels[spk_idx]

        return cls(event_labels, spike_labels_aligned, spike_times_aligned, window)

    @classmethod
    def from_series(
        cls,
        spike_timestamps: pd.Series,
        event_timestamps: pd.Series,
        window: Tuple[float, float],
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
        time_bins : array_like or None
            if provided, use these time bins to compute the PSTH
        time_step : float or None
            if provided, time bins are generated in the range of self.window with this step size

        Returns
        -------
        psth : xarray.DataArray
            PSTH matrix with dimensions 'trialid', 'unitid', 'time'

        Raises
        ------
        ValueError
            If neither time_bins nor time_step is provided
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
        psth, _ = np.histogramdd(
            np.array([trialids_digitized, unitids_digitized, self.spike_times]).T,
            bins=[
                np.arange(trialids_unique.size + 1),
                np.arange(unitids_unique.size + 1),
                time_bin_array,
            ],
        )
        time = time_bin_array[:-1] + (time_bin_array[1] - time_bin_array[0]) / 2
        psth = xr.DataArray(
            psth,
            coords={"trialid": trialids_unique, "unitid": unitids_unique, "time": time},
            dims=["trialid", "unitid", "time"],
        )
        return psth

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
            # save self.spike_times, self.trialids and self.unitids
            # as npy files within this archive
            with zf.open("spike_times.npy", "w") as f:
                np.save(f, self.spike_times)
            with zf.open("trialids.npy", "w") as f:
                np.save(f, self.trialids)
            with zf.open("unitids.npy", "w") as f:
                np.save(f, self.unitids)
            # save the window as a text file
            with zf.open("window.txt", "w") as f:
                value = f"{self.window[0]} {self.window[1]}"
                f.write(value.encode("utf-8"))

    @classmethod
    def load(cls, path: str) -> "SpikeTrainSet":
        with zipfile.ZipFile(path, "r") as zf:
            # load self.spike_times, self.trialids and self.unitids
            # from npy files within this archive
            with zf.open("spike_times.npy") as f:
                spike_times = np.load(f)
            with zf.open("trialids.npy") as f:
                trialids = np.load(f)
            with zf.open("unitids.npy") as f:
                unitids = np.load(f)
            # load the window from a text file
            with zf.open("window.txt") as f:
                window = tuple(map(float, f.read().decode("utf-8").split()))
                if len(window) != 2:
                    raise ValueError("Window must be a tuple of length 2")
        return cls(trialids, unitids, spike_times, window)

    def view(self):
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QApplication, QMainWindow
            from simianpy.analysis.spiketrain.spiketrainsetviewer import (
                SpikeTrainSetViewer,
            )
        except ImportError:
            raise ImportError(
                "pyqtmgl is not set up. Please install it to use this function."
            )
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        app = QApplication([])
        window = QMainWindow()
        window.setWindowTitle("Spike Train Set Viewer")

        sv = SpikeTrainSetViewer(self)
        window.setCentralWidget(sv)
        window.show()
        app.exec_()
