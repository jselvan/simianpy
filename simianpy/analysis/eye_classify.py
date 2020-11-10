import numpy as np

class EyeClassify:
    def __init__(self, eye_data, sampling_rate=1e3, filter_spec=None):
        if filter_spec is None:
            filter_spec={}

        self.eye_data = eye_data
        self.sampling_rate = sampling_rate

        self.velocity_data = self._diff(self.eye_data, filter_spec.get('velocity'))
        self.acceleration_data = self._diff(self.eye_data, filter_spec.get('acceleration'))

    def _diff(self, input_data, Filter=None):
        if Filter is None:
            diff = input_data.diff() * self.sampling_rate
        else:
            diff = input_data.apply(Filter).diff() * self.sampling_rate
        diff['radial'] = np.hypot(diff['eyeh'], diff['eyev'])
        return diff
    
    def detect_saccades(self, velocity_threshold, duration_threshold):
        pass

    def detect_fixations(self, velocity_threshold, duration_threshold):
        pass