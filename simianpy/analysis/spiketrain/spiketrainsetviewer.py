from PyQt5 import QtWidgets, QtCore
import numpy as np

# from pyqtmgl.widgets.continuous_viewer import ContinuousViewer
from pyqtmgl.widgets.graph import GraphWidget
class SpikeTrainSetViewer(QtWidgets.QWidget):
    def __init__(self, spiketrainset: "SpikeTrainSet"):
        super().__init__()
        self.spiketrainset = spiketrainset

        self.unique_unitids = np.unique(spiketrainset.unitids)
        self.unique_unitids_labels = [str(unitid) for unitid in self.unique_unitids]
        # we need a combobox listing all the unitids
        self.unitid_combobox = QtWidgets.QComboBox(self)
        self.unitid_combobox.addItems(self.unique_unitids_labels)
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
        self.time_step_spinbox.valueChanged.connect(self.compute)
        self.time_step_spinbox.setSuffix(" s")
        self.time_step_spinbox.setAlignment(QtCore.Qt.AlignRight)

        # if spiketrainset has trial_metadata, create a listbox with the trial metadata columns
        if self.spiketrainset.trial_metadata is not None:
            self.trial_metadata_listbox = QtWidgets.QListWidget(self)
            self.trial_metadata_listbox.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
            self.trial_metadata_listbox.setToolTip("Select trial metadata columns to display")
            for column in self.spiketrainset.trial_metadata.columns:
                item = QtWidgets.QListWidgetItem(column)
                item.setCheckState(QtCore.Qt.Unchecked)
                self.trial_metadata_listbox.addItem(item)
                # item.itemChanged.connect(self.compute)
            self.trial_metadata_listbox.itemChanged.connect(self.compute)
            self.trial_metadata_listbox.setMaximumHeight(200)
        else:
            self.trial_metadata_listbox = None


        # Layout: if epochs, use grid, else vertical
        if getattr(self.spiketrainset, 'epochids', None) is not None:
            self.has_epochs = True
            self.epoch_names = getattr(self.spiketrainset, 'epoch_names', None)
            if self.epoch_names is None:
                self.epoch_names = list(np.unique(self.spiketrainset.epochids))
            else:
                self.epoch_names = list(self.epoch_names)
            self.n_epochs = len(self.epoch_names)
            main_layout = QtWidgets.QVBoxLayout(self)
            main_layout.addWidget(self.unitid_combobox)
            main_layout.addWidget(self.time_step_spinbox)
            if self.trial_metadata_listbox is not None:
                main_layout.addWidget(self.trial_metadata_listbox)
            self.grid = QtWidgets.QGridLayout()
            self.rasterplots = []
            self.psthplots = []
            for i, epoch in enumerate(self.epoch_names):
                raster = GraphWidget()
                psth = GraphWidget()
                raster.add_scatter('raster', size=1)
                psth.add_line('psth', size=2)
                self.rasterplots.append(raster)
                self.psthplots.append(psth)
                label = QtWidgets.QLabel(str(epoch))
                label.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
                self.grid.addWidget(label, 0, i)
                self.grid.addWidget(raster, 1, i)
                self.grid.addWidget(psth, 2, i)
            main_layout.addLayout(self.grid)
        else:
            self.has_epochs = False
            main_layout = QtWidgets.QVBoxLayout(self)
            main_layout.addWidget(self.unitid_combobox)
            main_layout.addWidget(self.time_step_spinbox)
            if self.trial_metadata_listbox is not None:
                main_layout.addWidget(self.trial_metadata_listbox)
            self.rasterplots = [GraphWidget()]
            self.psthplots = [GraphWidget()]
            self.rasterplots[0].add_scatter('raster', size=1)
            self.psthplots[0].add_line('psth', size=2)
            main_layout.addWidget(self.rasterplots[0])
            main_layout.addWidget(self.psthplots[0])
        self.setLayout(main_layout)
        self.compute()

    def get_group_vars(self):
        """
        Get the group variables from the trial metadata listbox.
        If no listbox, return None.
        """
        if self.trial_metadata_listbox is not None:
            # selected_items = self.trial_metadata_listbox.selectedItems()
            selected_columns = [item.text() for item in self.trial_metadata_listbox.findItems("*", QtCore.Qt.MatchWildcard) if item.checkState() == QtCore.Qt.Checked]
            # selected_columns = [item.text() for item in selected_items]
            if selected_columns:
                return selected_columns
            else:
                return None
        else:
            return None

    def compute(self):
        timestep = self.time_step_spinbox.value()
        group = self.get_group_vars()
        self.spk, self.psth, self.cmap = self.spiketrainset.get_plotting_data(
            group=group,
            psth_params=dict(time_step=timestep),
        )
        for plot in self.psthplots:
            plot.nodes_by_name.clear() #TODO: implement direct clear method
            for group in self.cmap.keys():
                plot.add_line(group, size=2)
        self.psth = (self.psth/timestep).to_dataframe(name='psth')
        self.update_plot()

    def update_plot(self):
        unitid = self.unique_unitids[self.unitid_combobox.currentIndex()]
        epochs = self.epoch_names
        for plot in self.psthplots:
            plot.nodes_by_name.clear()
        if epochs is None:
            epochs = [None]
        for i, epoch in enumerate(epochs):
            rasterplot = self.rasterplots[i]
            psthplot = self.psthplots[i]
            if epoch is None:
                epochspk = self.spk.query("unitid==@unitid")
                epochpsth = self.psth.query("unitid == @unitid")
            else:
                epochspk = self.spk.query("unitid==@unitid & epoch == @epoch")
                epochpsth = self.psth.query("unitid == @unitid & epoch == @epoch")

            # Raster
            colors = np.array(epochspk.colors.values.tolist())
            self.rasterplots[i].update_node('raster',
                x=epochspk.spike_times,
                y=epochspk.trialid,
                colors=colors,
                alphas=1.
            )
            self.rasterplots[i].set_rect([
                self.spiketrainset.window[0],
                epochspk.trialid.min(),
                self.spiketrainset.window[1],
                epochspk.trialid.max()
            ])
            self.rasterplots[i].update()

            # PSTH 
            groupvars = self.get_group_vars()
            if groupvars is not None:
                grouped = epochpsth.groupby(groupvars)
            else:
                grouped = [('default', epochpsth)]
            for group, psth in grouped:
                psth = psth.reset_index()
                self.psthplots[i].add_line(group, size=2)
                self.psthplots[i].update_node(group,
                    x=psth.time,
                    y=psth.psth,
                    colors=self.cmap.get(group, [0, 0, 0]),
                    alphas=1.,
                    indices='auto'
                )
            self.psthplots[i].set_rect([
                self.spiketrainset.window[0],
                0,
                self.spiketrainset.window[1],
                epochpsth.values.max() if np.any(epochpsth.values) else 1
            ])
            self.psthplots[i].update()
    
def launch_viewer(spiketrainset: "SpikeTrainSet"):
    """
    launch a SpikeTrainSetViewer for the given SpikeTrainSet.
    """
    
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    app = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Spike Train Set Viewer")

    sv = SpikeTrainSetViewer(spiketrainset)
    window.setCentralWidget(sv)
    window.show()
    app.exec_()

