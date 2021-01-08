import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.figure_factory as ff


def qqplot(data, standardize=True, npoints=100, low=1e-1, high=0.9999,
           title='Normal Q-Q Plot', xaxis_title='Standard Normal Quantiles',
           yaxis_title='Sample quantiles'):
    """Creates a qq-plot in Plotly.

    Quantiles are generated inverting the cumulative distribuition function for
    the values in [low, high], equally spaced. The spacing is controlled by
    `npoints`.

    Parameters
    ----------
    data : numpy array
        Of shape (N,), containing the sample data.
    standardize : bool
        Whether `data` needs to be standardized.
    npoints : positive natural number
        The number of quantiles to plot.
    low : real number in (0, 1)
        The value of the cdf corresponding to the lowest quantile.
    high : real number in (0, 1)
        The value of the cdf corresponding to the highest quantile.
    title : str
        The plot title.
    xaxis_title : str
        The x axis title.
    yaxis_title : str
        The y axis title.

    Returns
    -------
    fig : plotly figure
        The figure containing the qq plot.

    """
    if standardize:
        data = (data - np.mean(data)) / np.std(data)

    # get quantiles
    where_quantiles = np.linspace(low, high, npoints)
    data_quantiles = np.quantile(data, where_quantiles)
    norm_quantiles = norm.ppf(where_quantiles)

    fig = go.Figure(go.Scatter(x=norm_quantiles, y=norm_quantiles,
                               mode='lines', name='Normal Quantiles'))
    fig.add_trace(go.Scatter(x=norm_quantiles, y=data_quantiles,
                             mode='markers',
                             name='Sample vs Normal Quantiles'))

    # final layout changes
    fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(title=yaxis_title)

    return fig


def distplot(data, standardize=True, title='Distplot', **kwargs):
    """Creates a distplot in Plotly, with the histogram, the KDE, a normal
    curve and a rug.

    Parameters
    ----------
    data : numpy array
        Of shape (N,), containing the sample data.
    standardize : bool
        Whether `data` needs to be standardized.
    title : str
        The plot title.
    The additional keyword arguments are for ff.create_distplot.

    Returns
    -------
    fig : plotly figure
        The figure containing the qq plot.

    """
    if standardize:
        data = (data - np.mean(data)) / np.std(data)

    fig = ff.create_distplot([data], ['distplot'], **kwargs)
    # can't create both kde and normal curves by default, so another figure
    # is built and the normal curve is taken from there
    fig2 = ff.create_distplot([data], ['distplot'], curve_type='normal',
                              **kwargs)
    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']
    fig.add_traces(go.Scatter(x=normal_x, y=normal_y, mode='lines',
                              name='Normal'))
    del fig2

    # final layout changes
    fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
    return fig
