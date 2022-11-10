import numpy as np
import plotly.graph_objects as go
from .utils import update_titles_


def plot_ucm_model(fit_result, title='', xaxis_title='', yaxis_title=''):
    observations = fit_result.data.endog
    if fit_result.data.dates is not None:
        index = fit_result.data.dates
    else:
        index = np.arange(len(observations))
    # plot original data
    fig = go.Figure(go.Scatter(x=index, y=observations, name='Observed'))
    level = fit_result.level.smoothed
    fig = fig.add_trace(go.Scatter(x=index, y=level, name='Level'))
    estimates = np.array(level)
    # plot level + seasonalities
    for seasonality in fit_result.freq_seasonal:
        name = seasonality.pretty_name
        seas_values = seasonality.smoothed
        estimates += seas_values
        fig.add_trace(go.Scatter(x=index, y=level+seas_values,
                                 name='Const + {}'.format(name)))
    # add estimates
    fig.add_trace(go.Scatter(x=index, y=estimates, mode='lines',
                             name='Estimates'))
    # final layout changes
    update_titles_(fig, title, xaxis_title, yaxis_title)
    return fig
