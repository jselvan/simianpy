import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np


class MultiPlotFigure:
    def __init__(self):
        self.plots = []  # To store plot objects

    def add_plot(self, plot):
        self.plots.append(plot)

    def show(self):
        # Determine the number of axes based on the number of plots
        num_axes = len(self.plots)
        if num_axes == 0:
            return

        fig, axes = plt.subplots(1, num_axes, figsize=(5 * num_axes, 5))

        for ax, plot in zip(axes, self.plots):
            plot.plot(fig, ax, self.highlight)

        plt.tight_layout()
        plt.show()

    def highlight(self, index):
        for plot in self.plots:
            plot.highlight(index)


class ScatterPlot:
    def __init__(self, x, y, label=None):
        self.x = x
        self.y = y
        self.label = label
        self.index = np.arange(len(x))  # Index for individual points

    def plot(self, fig, ax, highlight_callback):
        self.ax = ax
        self.highlight_callback = highlight_callback
        self.sc = ax.scatter(self.x, self.y, label=self.label)
        self.sc.set_picker(True)  # Enable pick events for points
        fig.canvas.mpl_connect("pick_event", self.on_pick)

    def on_pick(self, event):
        if isinstance(event.artist, PathCollection):
            ind = event.ind[0]
            self.highlight_callback(self.index[ind])

    def highlight(self, index):
        self.sc.set_sizes([30 if i == index else 10 for i in self.index])
        self.ax.figure.canvas.draw()


class LinePlotCollection:
    def __init__(self, x_values, y_values, labels=None):
        self.x_values = x_values
        self.y_values = y_values
        self.labels = labels
        self.index = np.arange(len(x_values))  # Index for individual lines

    def plot(self, fig, ax, highlight_callback):
        self.ax = ax
        self.highlight_callback = highlight_callback
        for x, y, label in zip(self.x_values, self.y_values, self.labels):
            (line,) = ax.plot(x, y, label=label)
            line.set_picker(True)  # Enable pick events for lines
        fig.canvas.mpl_connect("pick_event", self.on_pick)

    def on_pick(self, event):
        print(event.ind)
        if isinstance(event.artist, plt.Line2D):
            ind = self.index[event.ind[0]]
            self.highlight_callback(ind)

    def highlight(self, index):
        for line, i in zip(self.ax.lines, self.index):
            line.set_linewidth(2 if i == index else 1)
        self.ax.figure.canvas.draw()


# Example usage:
if __name__ == "__main__":
    figure = MultiPlotFigure()

    scatter_data = ScatterPlot([1, 2, 3, 4], [1, 4, 9, 16], label="Scatter Plot")

    line_data = LinePlotCollection(
        x_values=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        y_values=[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        labels=["Line 1", "Line 2", "Line 3"],
    )

    figure.add_plot(scatter_data)
    figure.add_plot(line_data)

    figure.show()
