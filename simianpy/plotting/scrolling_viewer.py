import simianpy as simi
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, CheckButtons
class ScrollingViewer:
    offset = 400
    start = 0
    view_size=int(6e4)
    bandpass_params = dict(order=3, highpass=300, lowpass=6000)
    def __init__(self, data, cmr=False, bandpass=False):
        self.data = data
        self.nsamples, self.nchannels = self.data.shape
        self.samples = np.arange(0, self.nsamples).astype(int)
        self.cmr = cmr
        self.tslice = slice(self.start, self.start+self.view_size)
        self.bandpass = bandpass
    def get_data(self):
        data = self.data[self.tslice, :]
        if self.cmr:
            data = data - np.median(data, axis=1, keepdims=True).astype(int)
        if self.bandpass:
            Filter = simi.signal.Filter(
                filter_type='bandpass', 
                filter_order=self.bandpass_params['order'], 
                freq_bounds=[
                    self.bandpass_params['highpass'], 
                    self.bandpass_params['lowpass']
                ], 
                sampling_frequency=3e4
            )
            for i in range(self.nchannels):
                data[:, i] = Filter(data[:, i])
        return data.T + (np.arange(self.nchannels)*self.offset)[:, None]
    def update_slider(self, val):
        self.tslice = slice(int(val), int(val+self.view_size))
        self.draw()
    def draw(self):
        t = self.samples[self.tslice]
        data = self.get_data()
        for line, y in zip(self.lines, data):
            line.set_data(np.stack([t, y]))
        self.ax.set_xlim(self.tslice.start, self.tslice.stop)
        self.fig.canvas.draw_idle()
    def check_clicked(self, label):
        if label=='cmr':
            self.cmr = not self.cmr
        elif label=='bandpass':
            self.bandpass = not self.bandpass
        self.draw()
    def init(self):
        t = self.samples[self.tslice]
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=.1, bottom=0.25)
        channels = np.arange(self.nchannels)
        self.lines = [self.ax.plot(t, row)[0] for row in self.get_data()]
        self.ax.set_ylim(-self.offset/2, self.nchannels*self.offset + self.offset/2)
        self.ax.set_yticks(channels*self.offset)
        self.ax.set_yticklabels(channels)

        ## SLIDER ##
        slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
        self.slider = Slider(
            ax=slider_ax, 
            label="Samples", 
            valmin=0, 
            valmax=self.nsamples, 
            valinit=self.start, 
            valstep=300,
            valfmt="%d"
        )
        self.slider.on_changed(self.update_slider)

        self.check_ax = plt.axes([0.01, 0.4, 0.05, 0.15])
        self.check = CheckButtons(self.check_ax, ['cmr','bandpass'], [self.cmr, self.bandpass])
        self.check.on_clicked(self.check_clicked)

        ## FORMATTING ##
        for spine in ['top', 'right']:
            self.ax.spines[spine].set_visible(False)
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Channel')
    def show(self):
        self.init()
        plt.show()

class ScrollingViewerMultiChannel(ScrollingViewer):
    """Scrolling Viewer but for when different channels are in separate arrays

    Parameters
    ----------
    data : [list of array-like]
    cmr: bool, default=False
    bandpass: bool, default=False
    """
    def __init__(self, data, cmr=False, bandpass=False):
        self.data = data
        self.nsamples = self.data[0].size
        self.nchannels = len(self.data)
        self.samples = np.arange(0, self.nsamples).astype(int)
        self.cmr = cmr
        self.bandpass = bandpass
        self.tslice = slice(self.start, self.start+self.view_size)
    def get_data(self):
        data = np.stack([row[self.tslice] for row in self.data])
        if self.cmr:
            data = data - np.median(data, axis=0, keepdims=True).astype(int)
        if self.bandpass:
            Filter = simi.signal.Filter(
                filter_type='bandpass', 
                filter_order=self.bandpass_params['order'], 
                freq_bounds=[
                    self.bandpass_params['highpass'], 
                    self.bandpass_params['lowpass']
                ], 
                sampling_frequency=3e4
            )
            for i in range(self.nchannels):
                data[i] = Filter(data[i])
        return data + (np.arange(self.nchannels)*self.offset)[:, None]