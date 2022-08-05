import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..File import File
from ..nex import Nex
from .openephys import load


class OpenEphys(File):
    """Interface for OpenEphys files

    Warnings
    --------
    This class is still under development!

    Handling of event data is currently hard coded into this class.
    Recommended use case is to subclass or monkey patch

    Parameters
    ----------
    filename: str or Path
    mode: str, optional, default: 'r'
        Must be one of ['r']
    recipe: str, Path or list
        recipe describing what files to load
    start_time: pd.Timestamp, optional, default: 0
        If you wish to provide a specific start time,
        provide a pandas Timestamp using pd.to_datetime
    use_cache: bool, optional, default: False
        If True, data will be dumped into an HDF file via h5py.File
    cache_path: Path, file-like object or None, optional, default: None
        If a cache_path is provided, the HDF file will be saved.
        Any valid input to h5py.File is accepted
        If None, a temporary file (tempfile.TemporaryFile) will be
        used. This file will be deleted upon closing the file
    overwrite_cache: bool, optional, default: False
        If false, data will not be loaded if already present in cache
        If true, data in cache will be overwritten
    logger: logging.Logger, optional
        logger for this object - see simi.io.File for more info

    Attributes
    ----------
    continuous_data
    spike_data
    event_data
    """

    description = """ """
    extension = [".continuous", ".spikes", ".events"]
    isdir = True
    needs_recipe = True
    default_mode = "r"
    modes = ["r"]
    supported_time_units = ["dt", "ms", "s"]

    def __init__(self, filename, **params):
        super().__init__(filename, **params)
        self.start_time = params.get("start_time", 0)

    def open(self):
        self._get_data_cache()
        if self.mode == "r":
            for file_params in self.recipe:
                filename = file_params["file"]
                filetype = file_params["type"]
                varname = file_params["name"]

                if (
                    not self.overwrite_cache
                    and filetype in self._data.keys()
                    and varname in self._data[filetype].keys()
                ):
                    continue
                if self.use_cache and filetype not in self._data.keys():
                    # unlike the defaultdict interface, h5py interface does not
                    # tolerate missing labels unless we use h5py address syntax
                    self._data.create_group(filetype)

                fpath = Path(self.filename, filename)
                if not fpath.is_file():
                    raise FileNotFoundError(
                        f"File ({fpath.name}) not found at {fpath.parent}"
                    )

                # header must be serialized to allow interoperability with hdf caching
                data = load(fpath, self.logger)
                header = data.pop("header")
                self._data[filetype][varname] = data
                self._data[filetype][varname]["header"] = json.dumps(header)

    def close(self):
        if self.mode == "r":
            pass

    def write(self, filename):
        raise NotImplementedError

    def read_timestamps(self, timestamps, start):
        if self.time_units == "dt":
            start = pd.to_datetime(start, format="%d-%b-%Y %H%M%S")
            return start + pd.to_timedelta(timestamps, unit="s")
        elif self.time_units == "ms":
            return timestamps * 1e3
        elif self.time_units == "s":
            return timestamps

    def _parse_continuous_data(self, cnt_data):
        header = json.loads(cnt_data["header"])
        block_length = int(header["blockLength"])
        sampling_rate = int(header["sampleRate"])
        start_time = header["date_created"]
        expanded_timestamps = (
            np.expand_dims(cnt_data["timestamps"], axis=1)
            + np.expand_dims(np.arange(block_length) / sampling_rate, axis=0)
        ).flatten()
        return pd.Series(
            cnt_data["data"],
            index=self.read_timestamps(
                timestamps=expanded_timestamps, start=start_time
            ),
        )

    def get_continuous_data(self, keys=None, resample_freq=None):
        """Get continuous data from openephys data as pandas dataframe

        Parameters
        ----------
        keys: list of str or None, optional, default: None
            subset of continuous data that will be retrieved
            if None, returns all data
        resample_freq: pd.DateOffset or str or None, optional, default: None
            valid time for new sample freq (e.g., '1L' or pd.offsets.Milli(1))
            if None, data is not resampled

        Returns
        -------
        continuous_data: pd.DataFrame
            columns will correspond to keys provided
            index will be a pd.DateTimeIndex using timestamps from openephys
        """
        if keys is None:
            keys = self._data["continuous"].keys()
        continuous_data = pd.DataFrame(
            {
                key: self._parse_continuous_data(self._data["continuous"][key])
                for key in keys
            }
        )
        if resample_freq is not None:
            continuous_data = continuous_data.asfreq(resample_freq)
        return continuous_data

    def _parse_spike_data(self, spk_data):
        header = json.loads(spk_data["header"])
        sample_in_microseconds = f"{1e6/float(header['sampleRate']):.3f}U"
        start_time = header["date_created"]
        return pd.DataFrame(
            spk_data["spikes"].squeeze(),
            columns=pd.timedelta_range(
                0, periods=spk_data["spikes"].shape[1], freq=sample_in_microseconds
            ),
            index=pd.MultiIndex.from_arrays(
                [
                    spk_data["sortedId"].squeeze(),
                    self.read_timestamps(
                        spk_data["timestamps"].squeeze(), start=start_time
                    ),
                ],
                names=("Unit", "Timestamp"),
            ),
        )

    def get_spike_data(self, keys=None):
        if keys is None:
            keys = self._data["spikes"].keys()
        spike_data = pd.concat(
            {key: self._parse_spike_data(self._data["spikes"][key]) for key in keys}
        )
        return spike_data

    def _parse_event_data(self, evt_data):
        header = json.loads(evt_data["header"])
        start_time = header["date_created"]
        vars = ["timestamps", "eventId", "channel"]
        evt_data_df = pd.DataFrame({var: evt_data[var].squeeze() for var in vars})
        evt_data_df["bitVal"] = 2 ** (7 - evt_data_df["channel"])
        event_data = (
            evt_data_df.query("eventId==1")
            .groupby("timestamps")
            .bitVal.sum()
            .astype(int)
        )
        event_data.index = self.read_timestamps(
            timestamps=event_data.index, start=start_time
        )
        return event_data

    def get_event_data(self, keys=None):
        if keys is None:
            keys = self._data["events"].keys()
        event_data = pd.DataFrame(
            {key: self._parse_event_data(self._data["events"][key]) for key in keys}
        )
        return event_data

    def to_nex(
        self,
        nexfile_path,
        timestampFrequency,
        PrethresholdTimeInSeconds=0.533,
        recipe=None,
        overwrite=False,
        **params,
    ):
        self.time_units = "s"
        nexfile_path = Path(nexfile_path)
        if nexfile_path.exists() and not overwrite:
            raise FileExistsError(f"{nexfile_path} already exists")

        if recipe is None:
            recipe = self.recipe
            self.logger.info(
                "No conversion recipe provided. Using OE recipe to convert all files."
            )

        start_time = time.time()
        self.logger.info(f'Converting to nex file "{nexfile_path}"')

        unit_as_char = lambda x: chr(int(x) - 1 + ord("a")) if x > 0 else "U"
        out_data = {
            "continuous": [],
            "spikes": [],
            "events": [],
        }
        self.logger.info("Preparing data specified in recipe")
        for file in recipe:
            self.logger.debug(f"Parsing {file}")
            header = json.loads(self._data[file["type"]][file["name"]]["header"])
            sampling_rate = int(header["sampleRate"])
            if file["type"] == "spikes":
                spike_data = self.get_spike_data([file["name"]])
                for unit, untidata in spike_data.groupby("Unit"):
                    waveforms = untidata.values
                    neuron_name = file["name"] + unit_as_char(unit)
                    out_data["spikes"].append(
                        {
                            "neuron_name": neuron_name,
                            "wave_name": neuron_name + "_wf",
                            "timestamps": untidata.index.get_level_values(
                                "Timestamp"
                            ).values,
                            "SamplingRate": sampling_rate,
                            "WaveformValues": waveforms,
                            "NPointsWave": waveforms.shape[1],
                            "PrethresholdTimeInSeconds": PrethresholdTimeInSeconds,
                            "wire": file["channel"],
                            "unit": int(unit),
                        }
                    )
            elif file["type"] == "continuous":
                continuous_data = self.get_continuous_data([file["name"]])[file["name"]]
                resample_to = file.get("resample_to", sampling_rate)
                if resample_to != sampling_rate:
                    dt = 1 / resample_to
                    timestamps = np.arange(
                        continuous_data.min(), continuous_data.max() + dt / 2, dt
                    )
                    values = np.interp(
                        timestamps, continuous_data.index.values, continuous_data.values
                    )
                    # continuous_data.reindex(timestamps).interpolate(
                    #     method="nearest", inplace=True
                    # )
                else:
                    values = continuous_data.values

                out_data["continuous"].append(
                    {
                        "name": file["name"],
                        "timestampOfFirstDataPoint": continuous_data.index[0],
                        "values": values,
                        "SamplingRate": resample_to,
                    }
                )
            elif file["type"] == "events":
                event_data = self.get_event_data([file["name"]])[file["name"]]
                name, fieldName = file["name"].split("/")
                out_data["events"].append(
                    {
                        "name": name,
                        "fieldNames": np.array([fieldName]),
                        "timestamps": event_data.index.values,
                        "markerFields": np.array([event_data.values]),
                    }
                )

        self.logger.info("Opening nex file to write data")
        with Nex(
            nexfile_path,
            mode="w",
            timestampFrequency=timestampFrequency,
            logger=self.logger,
            **params,
        ) as nexfile:
            self.logger.info("Writing continuous data")
            for continuous_data in out_data["continuous"]:
                nexfile.writer.AddContVarWithSingleFragment(**continuous_data)
            self.logger.info("Writing spike and waveform data")
            for spike_data in out_data["spikes"]:
                nexfile.writer.AddNeuron(
                    name=spike_data["neuron_name"],
                    timestamps=spike_data["timestamps"],
                    wire=spike_data["wire"],
                    unit=spike_data["unit"],
                )
                nexfile.writer.AddWave(
                    name=spike_data["wave_name"],
                    timestamps=spike_data["timestamps"],
                    SamplingRate=spike_data["SamplingRate"],
                    WaveformValues=spike_data["WaveformValues"],
                    NPointsWave=spike_data["NPointsWave"],
                    PrethresholdTimeInSeconds=spike_data["PrethresholdTimeInSeconds"],
                    wire=spike_data["wire"],
                    unit=spike_data["unit"],
                )
            self.logger.info("Writing event data")
            for event_data in out_data["events"]:
                nexfile.writer.AddMarker(**event_data)
            self.logger.info("Writing nex file to disk...")

        self.logger.info(f"\nSuccessfully wrote nexfile at path: {nexfile_path}")
        self.logger.info(f"Total time: {(time.time() - start_time):.3f} seconds\n\n")
        return nexfile
