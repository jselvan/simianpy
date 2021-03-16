from simianpy.plotting.util import add_colorbar, get_ax

import pandas as pd 

class Image:
    def __init__(self, x, y, z, data=None, fillna=None, ax=None, colorbar='vertical', im_kwargs={}):
        self.ax = get_ax(ax)

        if data is None:
            data = pd.DataFrame({'x':x, 'y':y, 'z':z})
            x, y, z = 'x', 'y', 'z'    
        data = data.reset_index().set_index([y, x]).unstack()[z].sort_index(ascending=False)
        if fillna is not None:
            data.fillna(fillna)
        
        bounds = data.columns[0], data.columns[-1], data.index[-1], data.index[0]
        self.im = self.ax.imshow(data.values, extent=(bounds), **im_kwargs)

        if colorbar:
            if colorbar not in ['vertical', 'horizontal']:
                colorbar = 'vertical'
            self.cbar = add_colorbar(self.im, orientation=colorbar)
        else:
            self.cbar = None
        self.data = data
        
    #TODO: move the standard implementation to from_dataframe and make this
    #  implementation work from raw data?
