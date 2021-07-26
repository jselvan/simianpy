from simianpy.io.trodes.readtrodes import readTrodesExtractedDataFile
from simianpy.io.File import File

from pathlib import Path

import numpy as np
from tqdm import tqdm

class Trodes(File):
    """Interface for SpikeGadgets Trodes extracted files

    Parameters
    ----------
    filename: str or Path
    session_name: str
    mmap: bool, optional, default: False
        Memory maps files if True
    pbar: bool, optional, default: False
        Will show tqdm progress bars if True
    mode: str, optional, default: 'r'
        Must be one of ['r']
    logger: logging.Logger, optional
        logger for this object - see simi.io.File for more info
    
    Attributes
    ----------
    vartypes_dic
    vartypes_dict_rev

    data
    vartypes
    varnames
    fileLength
    start_time

    continuous_data
    spike_data
    event_data
    """
    default_mode = 'r'
    modes = ['r']
    needs_recipe = True
    isdir = True

    def __init__(self, filename, session_name, mmap=False, pbar=False, **params):
        super().__init__(filename, **params)
        self.session_name = session_name
        self.mmap = mmap
        self.pbar = pbar
    
    def open(self):
        self._data = {}
        for name, info in self.recipe.items():
            datatype = info['type']
            if datatype == 'analog':
                self.logger.info(f'Reading analog data: {name}')
                timestamps, data = self._read_analog_data(name, info)
                self._data[name] = {'timestamps':timestamps, 'data':data}
            elif datatype == 'DIO':
                self.logger.info(f'Reading DIO data: {name}')
                on, off = self._read_dio_data(name, info)
                self._data[name] = {'on':on, 'off':off}
            else:
                raise ValueError(f"provided type {datatype} for {name} is not supported")
    
    def close(self):
        pass

    def _read_analog_data(self, name, info):
        mmap_mode = self.mode if self.mmap else 'r'
        timestamp_file = self.filename/info['timestamps_path'].format(name=self.session_name)
        data = {}
        timestamps = readTrodesExtractedDataFile(timestamp_file, mmap_mode=mmap_mode)[1]['time']

        channels = info['channels']

        if self.pbar:
            channels = tqdm(channels, desc=f'Reading Analog Data ({name})')

        for channel in channels:
            if channel is None:
                continue
            channel_file = self.filename/info['file_template_str']\
                .format(name=self.session_name, channel=channel)
            _, channeldata = readTrodesExtractedDataFile(channel_file, mmap_mode=mmap_mode)
            data[channel] = channeldata['voltage'].squeeze()

        return timestamps, data

    def _read_dio_data(self, name, info):
        mmap_mode = self.mode if self.mmap else 'r'
        filepath = self.filename/info['file_template_str'].format(name=self.session_name)
        _, dio = readTrodesExtractedDataFile(filepath, mmap_mode=mmap_mode)
        on, off = dio['time'][dio['state']==1], dio['time'][dio['state']==0]
        return on, off

    def dump(self, dumpdir, chunksize=1e7, end=None): #should chunksize be an attribute?
        chunksize = int(chunksize)
        dumpdir = Path(dumpdir)
        if not dumpdir.is_dir():
            dumpdir.mkdir()
        for name, info in self.recipe.items():
            datatype = info['type']
            if datatype == 'analog':
                self.logger.info(f'Dumping analog data: {name}')
                timestamps = self._data[name]['timestamps'][slice(None, end)]
                self.logger.info('Dumping timestamps file')
                np.save(dumpdir/f"{name}.timestamps.npy", timestamps)

                channels = self._data[name]['data'].keys()
                self.logger.info('Dumping channels file')
                with open(dumpdir/f"{name}.channels.txt", 'w') as f:
                    f.write("\n".join(map(str, channels)))

                self.logger.info('Initializing data file')
                #TODO: replace with just writing the header?
                datadumppath = dumpdir/f"{name}.npy"
                data = np.lib.format.open_memmap(
                    datadumppath, 
                    'w+', 
                    'int16', 
                    (len(timestamps), len(channels))
                )
                del data

                nchunks = np.ceil(timestamps.size/chunksize).astype(int)
                chunks = range(nchunks)
                if self.pbar:
                    chunks = tqdm(chunks, desc=f'Dumping Analog Data ({name}) in chunks')
                
                self.logger.info('Dumping data to file')
                with open(datadumppath, 'rb+') as f:
                    f.seek(128) #offset of numpy header
                    for i in chunks:
                        chunkslice = slice(i*chunksize, (i+1)*chunksize)
                        np.stack([
                            self._data[name]['data'][channel][chunkslice]
                            for channel in channels
                        ]).T.tofile(f)
                
                self.logger.info(f'Done dumping analog data: {name}!')
                # OLDER METHOD USING MEMMAP
                # for i in chunks:
                    # chunkslice = slice(i*chunksize, (i+1)*chunksize)
                #     # data = np.load(datadumppath, mmap_mode='r+')
                #     data[chunkslice, :] = np.stack([
                #         self._data[name]['data'][channel][chunkslice]
                #         for channel in channels
                #     ]).T
                #     # del data
                #     data.flush()
            elif datatype == 'DIO':
                self.logger.info(f'Dumping DIO: {name}')
                np.save(dumpdir/f"{name}.on.npy", self._data[name]['on'])
                np.save(dumpdir/f"{name}.off.npy", self._data[name]['off'])
            else:
                raise ValueError(f"provided type {datatype} for {name} is not supported")