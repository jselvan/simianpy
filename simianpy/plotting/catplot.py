import numpy as np
import pandas as pd


def get_agg_data(grouped_data, agg_method):
    if agg_method == "mean":
        agg_data = grouped_data.mean()
    elif agg_method == "median":
        agg_data = grouped_data.median()
    else:
        raise NotImplementedError()

    return agg_data


def get_error(grouped_data, error_method):
    if error_method is None or not error_method:
        return None
    else:
        if error_method.lower() == "se":
            error = grouped_data.apply(lambda x: x.std() / (x.count() ** 0.5))
        elif error_method.lower() == "ci95":
            error = grouped_data.apply(lambda x: 1.96 * x.std() / (x.count() ** 0.5))
        elif error_method.lower() == "std":
            error = grouped_data.std()
        elif error_method.lower() == "iqr":
            median = grouped_data.median()
            error = pd.DataFrame(
                {
                    "lower": median - grouped_data.quantile(0.25),
                    "upper": grouped_data.quantile(0.75) - median,
                }
            )
        else:
            raise NotImplementedError()

    return error


class CatPlot:
    def __init__(
        self,
        data,
        agg,
        cluster_var=None,
        axes_var=None,
        stack_var=None,
        agg_method="mean",
        error_method="se",
        x_offset=None,
        ax=None,
        clusters=None,
        cluster_params={},
        stack_params={},
        swarm=False,
        swarm_params={},
        swarm_cluster_params={},
        swarm_stack_params={},
        swarm_hist_params={},
        violin=False,
    ):
        self.agg = agg
        self.agg_method = agg_method
        self.error_method = error_method

        self.axes_var = axes_var
        self.cluster_var = cluster_var
        self.stack_var = stack_var

        self.cluster_params = cluster_params
        self.stack_params = stack_params
        self.swarm_params = swarm_params
        self.swarm_cluster_params = swarm_cluster_params
        self.swarm_stack_params = swarm_stack_params
        self.swarm_hist_params = swarm_hist_params

        self.swarm = swarm
        self.violin = violin

        self.xticklabels = list(data.groupby(self.agg).groups.keys())
        self.nbars = len(self.xticklabels)
        if self.cluster_var is None:
            self.clusters = (None,)
        else:
            self.clusters = clusters or list(
                data.groupby(self.cluster_var).groups.keys()
            )

        self.nclusters = len(self.clusters)
        self.x_offset = 1 / (self.nclusters + 1) if x_offset is None else x_offset

        self.xtick_coords = np.arange(self.nbars) + (self.nclusters / 2) * self.x_offset
        self.xtick_coords -= self.x_offset / 2

        if self.axes_var is None:
            self._plot_ax(data, ax)
        else:
            axes_grouped_data = data.groupby(self.axes_var)
            if isinstance(ax, dict):
                axes = ax
            else:
                indexes = list(axes_grouped_data.groups.keys())
                axes = dict(zip(indexes, ax))
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
            self._plot_stack(
                data, ax, cluster_name=cluster_name, offset=offset, bottoms=bottoms
            )
        else:
            for stack, stack_data in data.groupby(self.stack_var):
                bottoms = self._plot_stack(
                    stack_data,
                    ax,
                    cluster_name=cluster_name,
                    offset=offset,
                    stack_name=stack,
                    bottoms=bottoms,
                )

    def _plot_stack(
        self, data, ax, cluster_name=None, offset=0, stack_name=None, bottoms=None
    ):
        if cluster_name is None and stack_name is None:
            label = None
        elif cluster_name is None:
            label = stack_name
        elif stack_name is None:
            label = cluster_name
        else:
            label = cluster_name, stack_name

        x = np.arange(self.nbars) + offset * self.x_offset
        y = get_agg_data(data.groupby(self.agg), self.agg_method)
        yerr = get_error(data.groupby(self.agg), self.error_method)
        if bottoms is None:
            bottoms = np.zeros(len(x))
        y = y.reindex(self.xticklabels).values
        if yerr is not None:
            yerr = yerr.reindex(self.xticklabels).values.T

        self._plot(
            ax=ax,
            x=x,
            y=y,
            yerr=yerr,
            label=label,
            bottoms=bottoms,
            cluster_name=cluster_name,
            stack_name=stack_name,
        )
        if self.swarm:
            self._swarmplot(
                ax=ax,
                data=data,
                x=x,
                label=label,
                bottoms=bottoms,
                cluster_name=cluster_name,
                stack_name=stack_name,
            )
        if self.violin:
            self._violinplot(
                ax=ax,
                data=data,
                x=x,
                label=label,
                bottoms=bottoms,
                cluster_name=cluster_name,
                stack_name=stack_name,
            )

        return bottoms + y

    def _violinplot(self, ax, data, x, label, bottoms, cluster_name, stack_name):
        grouped_data = {key: data for key, data in data.groupby(self.agg)}
        violin_data = []
        xcoords = []
        for xcoord, xlabel, bottom in zip(x, self.xticklabels, bottoms):
            if xlabel in grouped_data:
                violin_data.append(grouped_data[xlabel] + bottom)
                xcoords.append(xcoord)
        parts = ax.violinplot(
            violin_data, xcoords, widths=self.x_offset, showextrema=False
        )
        for name, part in parts.items():
            if name == "bodies":
                for pc in part:
                    pc.set_alpha(0.15)
            else:
                pc.set_alpha(0.15)

    def _swarmplot(self, ax, data, x, label, bottoms, cluster_name, stack_name):
        for (_, group_data), xcoord in zip(data.groupby(self.agg), x):
            y = group_data.values
            y.sort()
            xmin, xmax = xcoord - (0.5 * self.x_offset), xcoord + (0.5 * self.x_offset)
            counts, edges = np.histogram(y, **self.swarm_hist_params)
            x_ = np.concatenate(
                [np.linspace(xmin, xmax, count + 2)[1:-1] for count in counts]
            )
            bin_centers = np.mean([edges[1:], edges[:-1]], axis=0)
            ax.scatter(
                x_,
                y,
                label=label,
                **self.swarm_params,
                **self.swarm_cluster_params.get(cluster_name, {}),
                **self.swarm_stack_params.get(stack_name, {})
            )

    def _plot(self, ax, x, y, yerr, label, bottoms, cluster_name, stack_name):
        return NotImplemented


class Bar(CatPlot):
    def __init__(self, **kwargs):
        if "bar_width" in kwargs:
            kwargs["x_offset"] = kwargs.pop("bar_width")
        super().__init__(**kwargs)

    def _plot(self, ax, x, y, yerr, label, bottoms, cluster_name, stack_name):
        ax.bar(
            x,
            y,
            yerr=yerr,
            label=label,
            width=self.x_offset,
            bottom=bottoms,
            **self.cluster_params.get(cluster_name, {}),
            **self.stack_params.get(stack_name, {})
        )


class Line(CatPlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _plot(self, ax, x, y, yerr, label, bottoms, cluster_name, stack_name):
        ax.errorbar(
            x,
            y + bottoms,
            yerr=yerr,
            label=label,
            **self.cluster_params.get(cluster_name, {}),
            **self.stack_params.get(stack_name, {})
        )


class ViolinPlot(CatPlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _plot(self, ax, x, y, yerr, label, bottoms, cluster_name, stack_name):
        ax.violinplot()
