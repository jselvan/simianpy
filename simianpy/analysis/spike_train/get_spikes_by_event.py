def get_spikes_by_event(spike_timestamps, event_timestamp, pad):
    return (
        spike_timestamps[
            ((event_timestamp + pad[0]) < spike_timestamps)
            & ((event_timestamp + pad[1]) > spike_timestamps)
        ]
        - event_timestamp
    )

