import numpy as np
import pandas as pd 

def get_agg_data(grouped_data, agg_method):
    if agg_method == 'mean':
        agg_data = grouped_data.mean()
    elif agg_method == 'median':
        agg_data = grouped_data.median()
    else:
        raise NotImplementedError()

    return agg_data

def get_error(grouped_data, error_method):
    if error_method is None or not error_method:
        return None
    else:
        if error_method.lower() == 'se':
            error = grouped_data.apply(lambda x: x.std()/(x.count()**.5))
        elif error_method.lower() == 'std':
            error = grouped_data.std()
        elif error_method.lower() == 'iqr':
            median = grouped_data.median()
            error = pd.DataFrame({'lower':median-grouped_data.quantile(.25), 'upper':grouped_data.quantile(.75) - median})
            # print(error, grouped_data.quantile([.25,.5,.75]))
        else:
            raise NotImplementedError()
    
    return error

class Bar:
    def __init__(self, data, agg, cluster_var=None, axes_var=None, stack_var=None, agg_method='mean', error_method='se', bar_width=None, ax=None, cluster_params={}, stack_params={}):
        self.agg = agg
        self.agg_method = agg_method
        self.error_method = error_method

        self.axes_var = axes_var
        self.cluster_var = cluster_var
        self.stack_var = stack_var

        self.cluster_params = cluster_params
        self.stack_params = stack_params

        self.xticklabels = list(data.groupby(self.agg).groups.keys())
        self.nbars = len(self.xticklabels)
        if self.cluster_var is None:
            self.clusters = None,
        else:
            self.clusters = list(data.groupby(self.cluster_var).groups.keys())
        
        self.nclusters = len(self.clusters)
        self.bar_width = 1/(self.nclusters+1) if bar_width is None else bar_width

        self.xtick_coords = np.arange(self.nbars) + (self.nclusters/2) * self.bar_width
        # if self.nclusters % 2 == 0:
        self.xtick_coords -= (self.bar_width/2)

        if self.axes_var is None:
            self._plot_ax(data,ax)
        else:
            axes_grouped_data = data.groupby(self.axes_var)
            if isinstance(ax,dict):
                axes = ax
            else:
                indexes = list(axes_grouped_data.groups.keys())
                axes = dict(zip(indexes,ax))
            for idx, ax_data in axes_grouped_data:
                self._plot_ax(ax_data, axes[idx], title=idx)
    
    def _plot_ax(self, data, ax, title=None):
        if self.cluster_var is None:
            self._plot_cluster(data, ax)
        else:
            grouped_data = data.groupby(self.cluster_var)
            for i, cluster in enumerate(self.clusters):
                # if cluster not in grouped_data.groups:
                #     continue
                try:
                    cluster_data = grouped_data.get_group(cluster)
                except KeyError:
                    continue
                self._plot_cluster(cluster_data, ax, cluster_name=cluster, offset=i)

        ax.set_xticks(self.xtick_coords)
        ax.set_xticklabels(self.xticklabels)

        ax.set_title(title)
    
    def _plot_cluster(self, data, ax, cluster_name=None, offset=0):
        bottoms = None
        if self.stack_var is None:
            self._plot_stack(data, ax, cluster_name=cluster_name, offset=offset, bottoms=bottoms)
        else:
            for stack, stack_data in data.groupby(self.stack_var):
                bottoms = self._plot_stack(stack_data, ax, cluster_name=cluster_name, offset=offset, stack_name=stack, bottoms=bottoms)
    
    def _plot_stack(self, data, ax, cluster_name=None, offset=0, stack_name=None, bottoms=None):
        if cluster_name is None and stack_name is None:
            label = None
        elif cluster_name is None:
            label = stack_name
        elif stack_name is None:
            label = cluster_name
        else:
            label = cluster_name, stack_name

        
        # x_offset = offset*self.bar_width
        # x = [self.xticklabels.index(x)+x_offset for x in y.index]
        x = np.arange(self.nbars) + offset*self.bar_width
        y = get_agg_data(data.groupby(self.agg), self.agg_method)
        yerr = get_error(data.groupby(self.agg), self.error_method)
        if bottoms is None:
            bottoms = np.zeros(len(x))
        y = y.reindex(self.xticklabels).values
        if yerr is not None:
            yerr = yerr.reindex(self.xticklabels).values.T

        # if bottoms is None:
        #     bottoms = np.zeros(len(x))
        # else:
        #     bottoms = [bottoms[self.xticklabels.index(x)] for x in y.index]

        ax.bar(x, y, yerr=yerr, label=label, width=self.bar_width, bottom=bottoms, **self.cluster_params.get(cluster_name,{}), **self.stack_params.get(stack_name,{}))
        return bottoms + y