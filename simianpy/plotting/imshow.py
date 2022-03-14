import numpy as np
import pandas as pd

from simianpy.plotting.util import add_colorbar, get_ax


class Image:
    def __init__(
        self,
        x,
        y,
        z,
        data=None,
        fillna=None,
        ax=None,
        colorbar="vertical",
        im_kwargs={},
    ):
        self.ax = get_ax(ax)

        if data is None:
            data = pd.DataFrame({"x": x, "y": y, "z": z})
            x, y, z = "x", "y", "z"
        data = (
            data.reset_index()
            .set_index([y, x])
            .unstack(x)[z]
            .sort_index(ascending=False)
        )
        if fillna is not None:
            data.fillna(fillna)

        bounds = data.columns[0], data.columns[-1], data.index[-1], data.index[0]
        self.im = self.ax.imshow(data.values, extent=(bounds), **im_kwargs)

        if colorbar:
            if colorbar not in ["vertical", "horizontal"]:
                colorbar = "vertical"
            self.cbar = add_colorbar(self.im, orientation=colorbar)
        else:
            self.cbar = None
        self.data = data

    @classmethod
    def from_matrix(
        cls, data, x=None, y=None, xlabel="x", ylabel="y", zlabel="z", **kwargs
    ):
        if x is None:
            x = np.arange(data.shape[1])
        if y is None:
            y = np.arange(data.shape[0])
        return cls(x, y, z=zlabel, data=data, xlabel=xlabel, ylabel=ylabel, **kwargs)

    # TODO: move the standard implementation to from_dataframe and make this
    #  implementation work from raw data?
