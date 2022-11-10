import numpy as np
import plotly.graph_objects as go
from .utils import update_titles_


def plot_exog(fitted_mod, indices=None, title='', xaxis_title='',
              yaxis_title=''):
    """Plots the result of the regression part of the model.

    Parameters
    ----------
    fitted_mod : suitable fitted statsmodels model
        The fitted model.
    indices : iterable of integers or None
        The indices indicating which exogenous variables to plot.
    title : str
        The title of the figure.
    xaxis_title : str
        The title of the x axis.
    yaxis_title : str
        The title of the y axis.

    Returns
    -------
    fig : plotly figure
        The figure containing the plot.

    """
    # retreive Foruier coefficients and Fourier regressors
    exog = fitted_mod.data.exog[:, indices]
    fourier_coef = fitted_mod.params[indices].values
    # compute seasonality
    season = fourier_coef * exog
    season = season.sum(axis=1)
    # get time index, if it exists
    time_idx = fitted_mod.data.dates
    if time_idx is None:
        time_idx = np.arange(exog.shape[0])
    # build figure
    fig = go.Figure(go.Scatter(x=time_idx, y=season, mode='lines'))
    # final layout changes
    update_titles_(fig, title, xaxis_title, yaxis_title)
    return fig
