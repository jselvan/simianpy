import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class VideoEyeAnimation:
    def __init__(self, eyedata, videodata, fps, transformation=None, tstep=16.66, speed=1, tracelength=100, traceparams={}):
        self.eyedata = eyedata
        if transformation is not None:
            self.eyedata = self.eyedata * transformation['gain'] + transformation['offset']
        self.videodata = videodata
        self.fps = fps
        self.t = 0 # current time in ms
        self.tmax = len(videodata) * 1e3 // fps # total time in ms

        self.tracelength = tracelength
        self.traceparams = traceparams
        self.tstep = tstep
        self.speed = speed

        self.initialize()

    @property
    def current_frame(self):
        frameidx = np.floor(self.t * self.fps / 1e3).astype('int')
        return self.videodata[frameidx]
    
    def initialize(self):
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.current_frame)
        self.eyetrace, = self.ax.plot([], [], **self.traceparams)
        self.anim = animation.FuncAnimation(
            self.fig, self._update_view,
            frames=np.arange(0, self.tmax, self.tstep),
            interval=self.tstep/self.speed,
            # blit=True,
            repeat=False,
        )

    def _update_view(self, *args):
        self.im.set_data(self.current_frame)
        if self.tracelength is not None:
            idx = slice(self.t-self.tracelength, self.t)
        else:
            idx = self.eyedata.index.get_indexer([self.t], method='nearest')
        self.eyetrace.set_data(
            self.eyedata.loc[idx, 'eyeh'], 
            self.eyedata.loc[idx, 'eyev']
        )
        self.t += self.tstep
    
    def show(self):
        plt.show()
    
    def save(self, filename):
        self.anim.save(filename, fps=1e3/(self.tstep/self.speed))
    
