from simianpy.misc import binary_digitize

from collections import namedtuple

import numpy as np
import scipy.stats

def ISI(spike_train):
    isi = np.diff(spike_train)
    return isi

def CV(isi):
    isi = np.asarray(isi)
    cv = isi.std()/isi.mean()
    return cv

def LV(isi):
    isi = np.asarray(isi)
    lv = (3 * (isi.size-2)) * (((isi[:-1] - isi[1:])/(isi[:-1]+isi[1:]))**2).sum()
    return lv

def gamma(isi):
    fit = namedtuple('gamma_mle_fit',['alpha','loc','beta'])(*scipy.stats.gamma.fit(isi))
    return fit

def ccg(spike_train_a, spike_train_b):
    ccg_data = np.correlate(spike_train_a, spike_train_b, 'full')
    return ccg_data

def acg(spike_train):
    #TODO: Remove center peak
    acg_data = ccg(spike_train, spike_train)
    return acg_data

def drop_overlapping(bursts):
    bursts_filtered = []
    if bursts:
        bursts_filtered.append(bursts.pop(0))
        while bursts:
            next_burst = bursts.pop(0)
            if bursts_filtered[-1]['offset'] >= next_burst['onset']:
                bursts_filtered[-1]['offset'] = next_burst['offset']
            else:
                bursts_filtered.append(next_burst)
    return bursts_filtered

def poisson_surprise_burst_detection(spike_train, time_range=None, minimum_burst_spikes=3, criterion=.05):
    max_spikes_in_burst_to_evaluate = 10
    bursts = []
    spike_train = np.sort(spike_train)
    n_spikes = spike_train.size
    if minimum_burst_spikes < 2:
        raise ValueError('minimum_burst_spikes must be 2 or greater')
    if n_spikes <= minimum_burst_spikes:
        return bursts

    if time_range is None:
        start_time, end_time = spike_train[0], spike_train[-1]
    else:
        start_time, end_time = time_range
    duration = end_time-start_time
    average_rate = n_spikes/duration
    instant_rate = minimum_burst_spikes/(spike_train[(minimum_burst_spikes-1):]-spike_train[:-(minimum_burst_spikes-1)])

    def surprise(onset, offset):
        count = offset-onset-1
        time = spike_train[offset]-spike_train[onset]
        return -scipy.stats.poisson.logsf(count, time*average_rate)

    spike_idx = 0
    burst = {}
    while spike_idx < instant_rate.size:
        current_rate = instant_rate[spike_idx]
        if burst:
            surprise_new = surprise(burst['onset'], spike_idx)
            if burst['extend_forward']<max_spikes_in_burst_to_evaluate:
                if surprise_new > burst['extend_forward_surprise']:
                    burst['offset'] = spike_idx
                    burst['extend_forward'] = 0
                    burst['extend_forward_surprise'] = surprise_new
                else:
                    burst['extend_forward'] += 1
            else:
                onset = burst['onset']+1
                while onset<burst['offset']:
                    surprise_old = surprise_new
                    surprise_new = surprise(onset, burst['offset'])
                    if surprise_new < surprise_old:
                        onset += 1
                    else:
                        break
                burst['onset'] = onset
                burst['surprise'] = surprise_new
                if burst['surprise'] > -np.log(criterion) and burst['offset']-burst['onset'] > minimum_burst_spikes:
                    bursts.append(burst)
                    for field in ['extend_forward', 'extend_forward_surprise']: burst.pop(field)
                else:
                    spike_idx = burst['onset']
                burst = {}
        else:
            if current_rate >= average_rate/2:
                burst = dict(onset=spike_idx, offset=spike_idx+minimum_burst_spikes)
                burst['extend_forward']=burst['extend_forward_surprise']=0
                spike_idx += minimum_burst_spikes
        spike_idx += 1
    # evaluate last burst if it was missed
    if burst:
        if burst['offset'] >= spike_train.size:
            burst['offset'] = spike_train.size-1
        burst['surprise'] = surprise(burst['onset'], burst['offset'])
        if burst['surprise'] > -np.log(criterion) and burst['offset']-burst['onset'] > minimum_burst_spikes:
            bursts.append(burst)
            for field in ['extend_forward', 'extend_forward_surprise']: burst.pop(field)
    
    bursts = drop_overlapping(bursts)
    for burst in bursts:
        burst['onset'] = spike_train[burst['onset']]
        burst['offset'] = spike_train[burst['offset']]
    # #TODO: activation times?
    # # Compute the activitation times by computing non-bursting firing rate and 
    # # find overlapping activation times to combine
      
    # # COMPUTE AVERAGE FIRING RATE WHEN NOT BURSTING
    # burst_time = burst_spikes = 0
    # for burst in bursts:
    #     burst_time += spike_train[burst['offset']] - spike_train[burst['onset']]
    #     burst_spikes += burst['offset'] - burst['onset']
    # average_nonbursting_rate = (n_spikes-burst_spikes)/(duration-burst_time)
    # # FIND ACTIVATION TIMES
    # for burst_idx, burst in enumerate(bursts):
    #     if burst_idx > 0:
    #         first_spike_after_last_burst = bursts[burst_idx-1]['offset']
    #     else:
    #         first_spike_after_last_burst = 0
    #     for spike_idx in range(first_spike_after_last_burst, burst['onset']):
    #         t = abs(spike_idx[burst['onset']] - spike_train[spike_idx])
    #         num_spike = burst['onset'] - spike_idx + 1
    #         surprise_new = surprise(t,num_spike,average_nonbursting_rate)

    #         if surprise_new >= prob:
    #             activation['activation_onset'] = spike_idx
    #             break
    #     else:
    #         activation['activation_onset'] = activation['onset']
        
    #     if activation_idx < len(activations)-1:
    #         first_spike_before_next_burst = activations[activation_idx+1]['onset']
    #     else:
    #         first_spike_before_next_burst = n_spikes-1
    #     for spike_idx in range(first_spike_before_next_burst, activation['offset'], -1):
    #         t = abs(spike_train[spike_idx]-spike_train[activation['offset']])
    #         num_spike = spike_idx - activation['offset'] + 1
    #         surprise_new = surprise(t,num_spike,average_nonbursting_rate)
            
    #         if surprise_new >= prob:
    #             activation['activation_offset'] = spike_idx
    #             break
    #     else:
    #         activation['activation_offset'] = activation['offset']
    return bursts