import struct
from functools import reduce
import os
from os import PathLike
from typing import Optional, BinaryIO
import warnings

from tqdm import tqdm
import numpy as np

def read_variable(file: BinaryIO, pbar: Optional[tqdm] = None):
    if pbar is not None:
        pbar.update(file.tell() - pbar.n)
    # Read the length of the variable name
    name_length_dtype = 'Q'
    name_length_size = struct.calcsize(name_length_dtype)
    name_length = struct.unpack(name_length_dtype, file.read(name_length_size))[0]
    name = file.read(name_length).decode('utf-8')

    var_type_dtype = 'Q'
    var_type_size = struct.calcsize(var_type_dtype)
    var_type_length = struct.unpack(var_type_dtype, file.read(var_type_size))[0]
    var_type = file.read(var_type_length).decode('utf-8')
    
    var_dims_dtype = 'Q'
    var_dims_size = struct.calcsize(var_dims_dtype)
    var_dims = struct.unpack(var_dims_dtype, file.read(var_dims_size))[0]

    var_size_dtype = 'Q' * var_dims
    var_size_size = struct.calcsize(var_size_dtype)
    var_size = struct.unpack(var_size_dtype, file.read(var_size_size))

    flat_size = (s for s in var_size[::-1] if s != 1)

    n_values = reduce(lambda x, y: x * y, var_size)

    dtype_map = {
        'double': 'd',
        'uint64': 'Q',
        'int64': 'q',
        'single': 'f',
        'uint32': 'I',
        'int32': 'i',
        'uint16': 'H',
        'int16': 'h',
        'logical': 'B',
        'char': 'c',
    }

    if var_type in dtype_map:
        data_type = dtype_map[var_type] * n_values
        data_size = struct.calcsize(data_type)
        data = struct.unpack(data_type, file.read(data_size))
        if n_values == 1:
            return {name: data[0]}
        else:
            return {name: np.array(data).reshape(*flat_size)}
    elif var_type == 'char':
        data = file.read(n_values).decode('utf-8')
        return {name: data}
    elif var_type == 'struct':
        num_fields_dtype = 'Q'
        num_fields_size = struct.calcsize(num_fields_dtype)
        num_fields = struct.unpack(num_fields_dtype, file.read(num_fields_size))[0]
        struct_data = []
        for _ in range(n_values):
            field_data = {}
            for _ in range(num_fields):
                data = read_variable(file, pbar)
                field_data.update(data)
            struct_data.append(field_data)
        # print(struct_data)
        if n_values == 1:
            struct_data = struct_data[0]
        return {name: struct_data}
    elif var_type == 'cell':
        cell_data = []
        for _ in range(n_values):
            result = read_variable(file, pbar)
            result = list(result.values())[0]
            cell_data.append(result)
        if n_values == 1:
            cell_data = cell_data[0]
        return {name: cell_data}
    elif var_type == 'function_handle':
        _, var_type_size = struct.unpack('QQ', file.read(16))
        var_type = file.read(var_type_size).decode('utf-8')
        
        n_values, _ = struct.unpack('QQ', file.read(16))
        nchar, = struct.unpack('Q', file.read(8))
        data = file.read(nchar).decode('utf-8')

        return {name: data}
    else:
        raise ValueError(f"Unsupported variable type: {var_type}")

def read_bhv2_raw(filename: PathLike[str] | str, pbar: bool=True):
    with open(filename, 'rb') as file:
        data = {}
        if pbar:
            pbar_obj = tqdm(total=os.path.getsize(filename))
        else:
            pbar_obj = None
        while True:
            try:
                data.update(read_variable(file, pbar_obj))
            except struct.error:
                break
        return data

def read_bhv2(filename: PathLike[str] | str, logger=None, pbar: bool=True, include_user_vars: bool=True):
    data = read_bhv2_raw(filename, pbar)
    n_trials = int(data['TrialRecord']['CurrentTrialNumber'])
    for trial in range(n_trials):
        trialkey = f'Trial{trial+1}'
        if trialkey not in data:
            warnings.warn(f'Missing trial {trialkey} in {filename}')
            continue
        trial_data = data[trialkey]
        condition = int(trial_data['Condition'])
        eye = trial_data['AnalogData']['Eye']
        markers = trial_data['BehavioralCodes']['CodeNumbers'].astype(int)
        timestamps = trial_data['BehavioralCodes']['CodeTimes']
        trial_error = trial_data['TrialError']
        start_time = trial_data['AbsoluteTrialStartTime']
        trial_info = {
            'trialid': trial+1,
            'condition': condition,
            'start_time': start_time,
            'eye': eye,
            'time': start_time + np.arange(eye.shape[1]),
            'markers': markers,
            'timestamps': timestamps,
            'trial_error': trial_error,
        }
        if include_user_vars:
            trial_info['user_vars'] = {
                key: trial_data['UserVars'][key]
                for key in trial_data['UserVars']
                if key not in ['SkippedFrameTimeInfo']
            }
        yield trial_info

if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    bhv2_data = read_bhv2_raw(path)
    print(bhv2_data.keys())