# pylint: skip-file
# Michael Gibson 17 July 2015
# Modified Zeke Arneodo Dec 2017
# Modified Adrian Foy Sep 2018
# Modified Janahan Selvanayagam Jan 2019
from ...misc import getLogger
from .intanutil.read_header import read_header
from .intanutil.get_bytes_per_data_block import get_bytes_per_data_block
from .intanutil.notch_filter import notch_filter
from .intanutil.data_to_result import data_to_result

import sys, os, time
from pathlib import Path

import numpy as np

def read_data(filename, notch = False, logger = None):
    """
    Read Intan RHS format files

    Required arguments:
    filename (str or pathlib.Path) -- full path to the RHS file

    Optional arguments:
    logger (logger or None; default = None) -- used for printing to screen and logging in .log file.  If None, a logger is initialized with log file sharing a name with RHS file
    notch (bool; default = False) -- if True and if software notch filter was selected during recording, reapply notch filter to amplifier data. Note the implementation of the notch filter used here seems very performance intensive and may considerably slow file reading.
    """
    #TODO: Implement caching or memory mapping for opening of large files 
    tic = time.time() 

    filename = Path(filename)
    if logger is None:
        logger = getLogger(__name__, filename.with_suffix('.log'))
    
    logger.info(f'Loading RHS file ({filename.name})...')

    with open(filename, 'rb') as f:
        filesize = os.path.getsize(filename)
        header = read_header(f, logger)        

        logger.info(f"Found {header['num_amplifier_channels']} amplifier channel(s).")
        logger.info(f"Found {header['num_board_adc_channels']} board ADC channel(s).")
        logger.info(f"Found {header['num_board_dac_channels']} board DAC channel(s).")
        logger.info(f"Found {header['num_board_dig_in_channels']} board digital input channel(s).")
        logger.info(f"Found {header['num_board_dig_out_channels']} board digital output channel(s).")

        bytes_per_block = get_bytes_per_data_block(header)
        logger.info(f'{bytes_per_block} bytes per data block')
        
        bytes_remaining = filesize - f.tell()

        if bytes_remaining == 0:
            logger.warn(f"Header file contains no data.  Amplifiers were sampled at {header['sample_rate']/1e3:0.2f} kS/s.")
            return None

        if bytes_remaining % bytes_per_block != 0:
            error_msg = 'Something is wrong with file size: should have a whole number of data blocks'
            logger.error(error_msg)
            raise Exception(error_msg)

        # DATA_BLOCK_SIZE = 128
        num_data_blocks = int(bytes_remaining / bytes_per_block) 
        record_time = 128 * num_data_blocks / header['sample_rate']

        logger.info(f"File contains {record_time:0.3f} seconds of data.  Amplifiers were sampled at {header['sample_rate']/1e3:0.2f} kS/s.")

        # define the data type for the data based on what channels are present
        dtype = [('t', np.dtype('<i'), 128)]
        if header['num_amplifier_channels'] > 0:
            dtype.append(('amplifier_data', np.uint16, ( header['num_amplifier_channels'], 128 )))
            if header['dc_amplifier_data_saved']:
                dtype.append(('dc_amplifier_data', np.uint16, ( header['num_amplifier_channels'], 128 )))
            dtype.append(('stim_data_raw', np.uint16, ( header['num_amplifier_channels'], 128 )))
        
        if header['num_board_adc_channels'] > 0:
            dtype.append(('board_adc_data', np.uint16, ( header['num_board_adc_channels'], 128 )))
            
        if header['num_board_dac_channels'] > 0:
            dtype.append(('board_dac_data', np.uint16, ( header['num_board_dac_channels'], 128 )))
                    
        if header['num_board_dig_in_channels'] > 0:
            dtype.append(('board_dig_in_raw', np.uint16, 128))

        if header['num_board_dig_out_channels'] > 0:
            dtype.append(('board_dig_out_raw', np.uint16, 128))
        
        # read the data using dtype into a numpy struct array 
        logger.debug('Reading data from file...')
        temp_data = np.fromfile(f, dtype, num_data_blocks)
        bytes_remaining = filesize - f.tell()
        if bytes_remaining == 0:
            logger.debug('... reached end of file!')
        else:
            error_msg = 'Error: End of file not reached.'
            logger.error(error_msg) 
            raise Exception(error_msg)

    # Parse out the data and scale to appropriate units 
    logger.debug('Parsing data...')
    data = {}
    data['t'] = temp_data['t'].flatten()

    if header['num_amplifier_channels'] > 0:  
        logger.debug('Scaling amplifier data to microvolts...')                   
        data['amplifier_data'] = 0.195 * (np.concatenate(temp_data['amplifier_data'], axis = 1).astype(np.int32) - 2**15) # units = microvolts
        if header['dc_amplifier_data_saved']:
            logger.debug('Scaling dc amplifier data to volts...')
            data['dc_amplifier_data'] = -0.01923 * (np.concatenate(temp_data['dc_amplifier_data'], axis = 1).astype(np.int32) - 2**8) # units = volts
        
        logger.debug('Parsing stimulation data and scaling to microvolts...')
        stim_data_raw =  np.concatenate(temp_data['stim_data_raw'], axis = 1)    

        data['compliance_limit_data'] = (stim_data_raw & 2**15) != 0 # get 2^15 bit, interpret as True or False
        data['charge_recovery_data'] = (stim_data_raw & 2**14) != 0 # get 2^14 bit, interpret as True or False
        data['amp_settle_data'] = (stim_data_raw & 2**13) != 0  # get 2^13 bit, interpret as True or False
        
        stim_polarity = 1 - (2 * ((stim_data_raw & 2**8) != 0)) # get 2^8 bit, interpret as +1 for 0_bit or -1 for 1_bit
        curr_amp = stim_data_raw & (2**8 - 1) # get least-significant 8 bits corresponding to the current amplitude
        data['stim_data'] = header['stim_step_size'] * (curr_amp * stim_polarity / 1.0e-6) # multiply current amplitude by the correct sign
    
    if header['num_board_adc_channels'] > 0:
        logger.debug('Scaling board adc data to microvolts...')
        data['board_adc_data'] = 0.0003125 * (np.concatenate(temp_data['board_adc_data'], axis = 1).astype(np.int32) - 2**15) # units = microvolts

    if header['num_board_dac_channels'] > 0:
        logger.debug('Scaling board dac data to microvolts...')
        data['board_dac_data'] = 0.0003125 * (np.concatenate(temp_data['board_dac_data'], axis = 1).astype(np.int32) - 2**15) # units = microvolts

    if header['num_board_dig_in_channels'] > 0:
        logger.debug('Parsing digital input data...')
        data['board_dig_in_data'] = np.not_equal(
            np.bitwise_and(
                temp_data['board_dig_in_raw'].flatten()[None,:] ,
                1 << np.array([ch['native_order'] for ch in header['board_dig_in_channels']])[:, None]
            ), 
            0
        )

    if header['num_board_dig_out_channels'] > 0:
        logger.debug('Parsing digital output data...')
        data['board_dig_out_data'] = np.not_equal(
            np.bitwise_and(
                temp_data['board_dig_out_raw'].flatten()[None,:] ,
                1 << np.array([ch['native_order'] for ch in header['board_dig_out_channels']])[:, None]
            ), 
            0
        )

    logger.debug('Checking for gaps in timestamps...')
    gaps = np.diff(data['t']) != 1
    if not gaps.any():
        logger.info('No missing timestamps in data.')
    else:
        logger.warn(f'Warning: {gaps.sum()} gaps in timestamp data found.  Time scale will not be uniform!')

    logger.debug('Scaling timestamps using sampling rate...')
    data['t'] = data['t'] / header['sample_rate']
    
    if notch:
        logger.debug('Applying notch filter')
        if header['notch_filter_frequency'] > 0:
            for i in range(header['num_amplifier_channels']):
                data['amplifier_data'][i, :] = notch_filter(data['amplifier_data'][i, :], header['sample_rate'],
                                                            header['notch_filter_frequency'], 10)
    else:
        logger.debug('Skipping notch filter')

    logger.debug('Moving variables to result struct...')
    result = data_to_result(header, data, True)

    logger.info('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))
    return result

if __name__ == '__main__':
    a = read_data(sys.argv[1])
    #print(a)