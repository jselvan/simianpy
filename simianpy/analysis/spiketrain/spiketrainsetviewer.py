
from PyQt5 import QtWidgets, QtCore
import numpy as np

# from pyqtmgl.widgets.continuous_viewer import ContinuousViewer
from pyqtmgl.widgets.graph import GraphWidget
class SpikeTrainSetViewer(QtWidgets.QWidget):
    def __init__(self, spiketrainset: "SpikeTrainSet"):
        super().__init__()
        from pyqtmgl.widgets.continuous_viewer import ContinuousViewer
        from pyqtmgl.widgets.scatter import ScatterWidget
        self.spiketrainset = spiketrainset

        self.unique_unitids = np.unique(spiketrainset.unitids).astype(str)
        # we need a combobox listing all the unitids
        self.unitid_combobox = QtWidgets.QComboBox(self)
        self.unitid_combobox.addItems(self.unique_unitids)
        self.unitid_combobox.currentIndexChanged.connect(self.update_plot)
        self.unitid_combobox.setCurrentIndex(0)
        self.unitid_combobox.setToolTip("Select unit id to plot")
        self.unitid_combobox.setMinimumWidth(200)
        
        # we need a spinbox for the time step for the PSTH
        self.time_step_spinbox = QtWidgets.QDoubleSpinBox(self)
        self.time_step_spinbox.setRange(0.001, 1.0)
        self.time_step_spinbox.setSingleStep(0.001)
        self.time_step_spinbox.setDecimals(3)
        self.time_step_spinbox.setValue(0.01)
        self.time_step_spinbox.setToolTip("Time step for PSTH")
        self.time_step_spinbox.setMinimumWidth(200)
        self.time_step_spinbox.valueChanged.connect(self.update_plot)
        self.time_step_spinbox.setSuffix(" s")
        self.time_step_spinbox.setAlignment(QtCore.Qt.AlignRight)

        # create the layout, which includes the above widgets, 
        # a scatter plot for the raster and a line plot for the PSTH
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.unitid_combobox)
        self.layout.addWidget(self.time_step_spinbox)
        # self.layout.addStretch(1)
        self.rasterplot = GraphWidget()
        self.psthplot = GraphWidget()
        self.rasterplot.add_scatter(
            'raster',
            size=1
        )
        self.psthplot.add_line(
            'psth',
            size=2
        )
        self.layout.addWidget(self.rasterplot)
        self.layout.addWidget(self.psthplot)
        # self.psth = ContinuousViewer()
        # self.layout.addWidget(self.psth)
        self.setLayout(self.layout)
        self.update_plot()

    def update_plot(self):
        unitid = self.unique_unitids[self.unitid_combobox.currentIndex()]
        unitid = int(unitid)
        # get the spike times for this unit
        mask = self.spiketrainset.unitids == unitid
        spike_times = self.spiketrainset.spike_times[mask]
        trialids = self.spiketrainset.trialids[mask]
        # get the PSTH for this unit
        time_step = self.time_step_spinbox.value()
        psth = self.spiketrainset.to_psth(time_step=time_step).sel(unitid=unitid).mean(dim='trialid') / time_step
        # update the scatter plot
        self.rasterplot.update_node('raster',
            x=spike_times,
            y=trialids,
            colors=[0,0,0], # white,
            alphas=1.
        )
        self.psthplot.update_node('psth',
            x=psth.time,
            y=psth.values,
            colors=[0.5, 0.5, 0.5],
            alphas=1.,
            indices='auto'
        )
        self.rasterplot.set_rect(
            [self.spiketrainset.window[0],
            self.spiketrainset.trialids.min(),
            self.spiketrainset.window[1],
            self.spiketrainset.trialids.max()]
        )
        self.psthplot.set_rect(
            [self.spiketrainset.window[0],
            0,
            self.spiketrainset.window[1],
            psth.values.max()]
        )
        
        # update the PSTH plot
        # self.psth.set_data(x=psth.time, y=psth.values)
        self.rasterplot.update()
        self.psthplot.update()
