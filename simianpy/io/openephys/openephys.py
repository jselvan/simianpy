# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 15:18:38 2014

@author: Dan Denman and Josh Siegle

Modified by: Janahan Selvanayagam

Loads .continuous, .events, and .spikes files saved from the Open Ephys GUI

Usage:
    import OpenEphys
    data = OpenEphys.load(pathToFile) # returns a dict with data, timestamps, etc.

"""

import os
from pathlib import Path
import numpy as np

# constants
NUM_HEADER_BYTES = 1024
SAMPLES_PER_RECORD = 1024
BYTES_PER_SAMPLE = 2
RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD * BYTES_PER_SAMPLE + 10 # size of each continuous record in bytes
RECORD_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

def load(filepath, logger = None):
    filepath = Path(filepath)

    # redirects to code for individual file types
    load_dict = {
        '.continuous': loadContinuous,
        '.spikes': loadSpikes,
        '.events': loadEvents
    }
    if filepath.suffix in load_dict:
        data = load_dict[filepath.suffix](filepath, logger)
    else:
        raise Exception(f"Not a recognized file type. File suffix must be in: {load_dict.keys()}")
    data['timestamps'] /= int(data['header']['sampleRate'])
    return data

def loadContinuous(filepath, logger = None):
    if logger is None:
        from ...misc import getLogger 
        logger = getLogger(__name__)

    logger.debug(f'Loading continuous data ({os.path.basename(filepath)})...')

    with open(filepath, 'rb') as f:
        header = readHeader(f, logger)

        dtype = [('timestamps', np.dtype('<i8'),1), # little-endian 64-bit signed integer
        ('N', np.dtype('<u2'), 1), # little-endian 16-bit unsigned integer
        ('recordingNumbers', np.dtype('>u2'), 1), # big-endian 16-bit unsigned integer
        ('data', np.dtype('>i2'), SAMPLES_PER_RECORD), # big-endian 16-bit signed integer
        ('marker', np.dtype('b'), 10) #dump
        ]

        fileLength = os.fstat(f.fileno()).st_size - f.tell()
        recordSize = np.dtype(dtype).itemsize
        if fileLength % recordSize != 0:
            msg = "File size is not consistent with a continuous file: may be corrupt"
            logger.error(msg)
            raise Exception(msg)
    
        nrecords = fileLength//recordSize
        records = np.fromfile(f, dtype, nrecords)
        
    data = { }
    data['header'] = header
    data['timestamps'] = records['timestamps'].astype(float)
    data['recordingNumber'] = records['recordingNumbers'].astype(float)
    
    corrupt_records = (records['N'] != SAMPLES_PER_RECORD)
    if corrupt_records.any():
        msg = f'Found corrupted record(s) in block(s):\n {", ".join(np.where(corrupt_records))}'
        logger.error(msg)
        raise Exception(msg)

    data['data'] = records['data'].flatten()*float(header['bitVolts'])

    return data

def loadSpikes(filepath, logger = None):
    if logger is None:
        from ...misc import getLogger 
        logger = getLogger(__name__)

    logger.debug(f'Loading spikes ({os.path.basename(filepath)})...')

    with open(filepath, 'rb') as f:
        header = readHeader(f, logger)
        
        if float(header[' version']) < 0.4:
            msg = 'Loader is only compatible with .spikes files with version 0.4 or higher'
            logger.error(msg)
            raise Exception(msg)

        numChannels = int(header['num_channels'])
        numSamples = 40 # **NOT CURRENTLY WRITTEN TO HEADER**
        dtype = [('eventType', np.dtype('<u1'),1),
        ('timestamps', np.dtype('<i8'), 1),
        ('software_timestamp', np.dtype('<i8'), 1),
        ('source', np.dtype('<u2'), 1),
        ('numChannels', np.dtype('<u2'), 1),
        ('numSamples', np.dtype('<u2'), 1),
        ('sortedId', np.dtype('<u2'),1),
        ('electrodeId', np.dtype('<u2'),1),
        ('channel', np.dtype('<u2'),1),
        ('color', np.dtype('<u1'), 3),
        ('pcProj', np.float32, 2),
        ('sampleFreq', np.dtype('<u2'),1),
        ('waveforms', np.dtype('<u2'), numChannels*numSamples),
        ('gain', np.float32, numChannels),
        ('thresh', np.dtype('<u2'), numChannels),
        ('recNum', np.dtype('<u2'), 1)
        ]
        
        nrecords = (os.fstat(f.fileno()).st_size - f.tell())//np.dtype(dtype).itemsize
        records = np.fromfile(f, dtype, nrecords)

        data = {}
        data['header'] = header
        
        data['gain'] = records['gain'].reshape([nrecords, numChannels]).astype(float)
        data['thresh'] = records['thresh'].reshape([nrecords, numChannels]).astype(float)
        data['sortedId'] = records['sortedId'].reshape([nrecords, numChannels]).astype(float)

        data['spikes'] = records['waveforms'].reshape([nrecords,numSamples, numChannels]).astype(float)
        data['spikes'] -= 32768
        for ch in range(numChannels):
            data['spikes'][:,:,ch] = (data['spikes'][:,:,ch].T / data['gain'][:,ch]).T * 1000

        data['timestamps'] = records['timestamps'].astype(float)
        data['source'] = records['source'].astype(float)
        data['recordingNumber'] = records['recNum'].astype(float)
    return data
    
def loadEvents(filepath, logger = None):
    if logger is None:
        from ...misc import getLogger 
        logger = getLogger(__name__)

    logger.debug(f'Loading events ({os.path.basename(filepath)})...')

    with open(filepath, 'rb') as f:
        header = readHeader(f, logger)

        if float(header[' version']) < 0.4:
            msg = 'Loader is only compatible with .events files with version 0.4 or higher'
            logger.error(msg)
            raise Exception(msg)
        
        dtype = [('timestamps', np.dtype('<i8'),1),
        ('sampleNum', np.dtype('<i2'), 1),
        ('eventType', np.dtype('<u1'), 1),
        ('nodeId', np.dtype('<u1'), 1),
        ('eventId', np.dtype('<u1'), 1),
        ('channel', np.dtype('<u1'), 1),
        ('recordingNumber', np.dtype('<u2'), 1)
        ]

        fileLength = os.fstat(f.fileno()).st_size - f.tell()
        recordSize = np.dtype(dtype).itemsize
        if fileLength % recordSize != 0:
            msg = "File size is not consistent with an events file: may be corrupt"
            logger.error(msg)
            raise Exception(msg)
    
        nrecords = fileLength//recordSize
        records = np.fromfile(f, dtype, nrecords)

    data = {}
    data['header'] = header
    data['timestamps'] = records['timestamps'].astype(float)
    data['sampleNum'] = records['sampleNum'].astype(float)
    data['eventType'] = records['eventType'].astype(float)
    data['nodeId'] = records['nodeId'].astype(float)
    data['eventId'] = records['eventId'].astype(float)
    data['channel'] = records['channel'].astype(float)
    data['recordingNumber'] = records['recordingNumber'].astype(float)

    return data

def readHeader(f, logger = None):
    if logger is None:
        from ...misc import getLogger 
        logger = getLogger(__name__)
    
    logger.debug('Reading header...')

    header = { }
    h = f.read(1024).decode().replace('\n','').replace('header.','')
    for _,item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0]] = item.split(' = ')[1]
    
    logger.debug(header)
    return header