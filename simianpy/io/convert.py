"""tools to convert files

Currently implemented:
ephys2nex -- convert OpenEphys files to Neuroexplorer (nex) files
"""
import os
import time
import warnings
from itertools import groupby

import numpy as np
import scipy

from simianpy.io.nex.nexfile import NexWriter
from simianpy.io.openephys import load
from simianpy.misc import getLogger


def ephys2nex(
    ephys_path,
    nexfile_path,
    SamplingRate_spikes=3e4,
    SamplingRate_continuous=1e3,
    num_channels=96,
    NPointsWave=40,
    PrethresholdTimeInSeconds=0.533,
    spike_prefix="Sep107",
    LFP_prefix="100_CH",
    eye_channels={"eyeh": "100_ADC1.continuous", "eyev": "100_ADC2.continuous"},
    logger=None,
):
    """Loads OpenEphys data found at 'ephys_path' and outputs a '.nex' file at path 'nexfile_path'

    Parameters
    ----------
    ephys_path: str
        valid path to a folder containing OpenEphys data (i.e. contains files with extension '.continuous', '.spikes', '.events', etc.) to be loaded
    nexfile_path: str
        valid path & file name where nexfile_path will be saved. This function will not overwrite an existing file. Must end with extension '.nex'
    SamplingRate_spikes: int; default = 3e4
        sampling rate of .spikes files in Hz
    SamplingRate_continuous: int; default = 1e3
        sampling rate of .continuous files in Hz
    num_channels: int; default = 32
        number of channels
    NPointsWave: int; default = 40
        number of data points in each waveform in .spikes files
    PrethresholdTimeInSeconds: float; default = 0.533
        pre-threshold time in seconds
    spike_prefix: str; default = 'SEp115
        prefix of all spike file names
    LFP_prefix: str; default = '110_CH
        prefix of all LFP file names
    eye_channels: dict <str:str>; default = {'eyeh': "110_ADC1.continuous", 'eyev': "110_ADC2.continuous"
        dict containing file names for eyeh and eyev

    Returns
    -------
    None
    """
    start_time = time.time()

    if logger is None:
        logger = getLogger(__name__)

    # Some error handling
    if isinstance(ephys_path, str):
        if not os.path.isdir(ephys_path):
            raise ValueError(f"Provided ephys_path: {ephys_path} is not a valid folder")
    else:
        raise TypeError(
            f"Provided ephys_path: {ephys_path} is not a string. Please provide a path to a folder containing openephys data."
        )

    if isinstance(nexfile_path, str):
        if not nexfile_path.endswith(".nex"):
            raise ValueError(
                f"Output nexfile_path must end with extension '.nex'.  Provided nexfile_path was invalid: {nexfile_path}"
            )
        if os.path.isfile(nexfile_path):
            raise Exception(f"A file already exists at path {nexfile_path}")
    else:
        raise TypeError(
            f"Provided nexfile_path: {ephys_path} is not a string. Please provide a path where nexfile_path can be saved."
        )

    unit_as_char = lambda x: chr(x - 1 + ord("a")) if x > 0 else "U"
    writer = NexWriter(SamplingRate_spikes, useNumpy=True)

    # add data
    for i in range(num_channels):
        logger.info("\nFor channel %d:" % (i + 1))
        # load spike data
        spike_fpath = os.path.join(ephys_path, f"{spike_prefix}.0n{i}.spikes")
        spike_data = load(spike_fpath, logger)

        units = np.unique(spike_data["sortedId"])
        for unit_num, unit_id in enumerate(units):
            if spike_data["spikes"].shape[1] != NPointsWave:
                raise ValueError(
                    f"The spikes file at the following path has the wrong number of NPointsWave. \n fpath: {spike_fpath}"
                )

            unit_name = unit_as_char(unit_num)
            neuron_name = f"sig{i + 1:03d}{unit_name}"
            wave_name = f"{neuron_name}_wf"

            idx = spike_data["sortedId"].squeeze() == unit_id

            neuronTs = spike_data["timestamps"][idx].squeeze()
            WaveformValues = spike_data["spikes"][idx]

            try:
                while WaveformValues.ndim > 3:
                    WaveformValues = WaveformValues.squeeze(axis=0)
            except:
                warnings.warn(
                    f"Failed to shape WaveformValues appropriately. Skipping unit: {wave_name}"
                )
                continue

            if WaveformValues.shape[1] != NPointsWave:
                warnings.warn(
                    f"Waveforms for unit {wave_name} has {WaveformValues.shape[1]} points instead of {NPointsWave} points as specified by arg NPointsWave. NPointsWave will be adjusted for this unit - there may be unintended consequences."
                )

            # add neuron & spike waveforms
            writer.AddNeuron(name=neuron_name, timestamps=neuronTs)
            writer.AddWave(
                name=wave_name,
                timestamps=neuronTs,
                SamplingRate=SamplingRate_spikes,
                WaveformValues=WaveformValues,
                NPointsWave=NPointsWave,
                PrethresholdTimeInSeconds=PrethresholdTimeInSeconds,
                wire=i,
                unit=unit_num,
            )

        # #add continuous data
        # continuous_fpath = os.path.join(ephys_path, f"{LFP_prefix}{i + 1}.continuous")
        # continuous_data = load(continuous_fpath, logger = logger)

        # AD_name = f"AD{i + 1:02d}"

        # #decimates by factor 30, using Chebyshev type I infinite impulse response filter of order 8 (in theory this is the same as MATLAB decimate)
        # writer.AddContVarWithSingleFragment(name = AD_name,
        # timestampOfFirstDataPoint = continuous_data['timestamps'][0],
        # SamplingRate = SamplingRate_continuous,
        # values = scipy.signal.decimate(continuous_data['data'], 30)
        # )

    # add eye channels
    logger.info("\nFor eye channel:")
    for eye_channel, fname in eye_channels.items():
        continuous_fpath = os.path.join(ephys_path, fname)
        continuous_data = load(continuous_fpath, logger)

        # decimates by factor 30, using Chebyshev type I infinite impulse response filter of order 8 (in theory this is the same as MATLAB decimate)
        writer.AddContVarWithSingleFragment(
            name=eye_channel,
            timestampOfFirstDataPoint=continuous_data["timestamps"][0],
            SamplingRate=SamplingRate_continuous,
            values=scipy.signal.decimate(continuous_data["data"], 30),
        )

    # add event codes
    logger.info("\nFor events:")
    event_fpath = os.path.join(ephys_path, "all_channels.events")
    event_data = load(event_fpath, logger)

    data = np.stack(
        [event_data["timestamps"], event_data["eventId"], 2**(7-event_data["channel"])]
    ).T.squeeze()
    timestamps = []
    markers = []
    for timestamp, values in groupby(data, lambda x: x[0]):
        _, state, bits = list(zip(*values))
        if state[0]:
            markers.append(f"{int(sum(bits)):03d}")
            timestamps.append(timestamp)
    markers = np.array([markers])
    timestamps = np.array(timestamps)

    # no clue why this is done but it was in MATLAB code
    # for i in range(len(markers) - 4):
    #     if markers[i:(i+5)] == [1,2,4,8,16]:
    #         markers[i:(i+5)] = [300]*5

    writer.AddMarker(
        name="Strobed",
        timestamps=timestamps,
        fieldNames=np.array(["DIO"]),
        markerFields=markers,
    )
    writer.WriteNexFile(nexfile_path)

    logger.info(f"\nSuccessfully wrote nexfile at path: {nexfile_path}")
    logger.info(f"Total time: {(time.time() - start_time):.3f} seconds\n\n")