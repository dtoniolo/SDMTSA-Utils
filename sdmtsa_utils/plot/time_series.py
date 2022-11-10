import numpy as np
import pandas as pd
import plotly
import plotly.subplots
import plotly.graph_objects as go
from statsmodels.tsa import stattools
import scipy
import scipy.stats


def plot_ts(timeseries, lags, alpha=0.05, ci=False, title='', xaxis_title='',
            yaxis_title=''):
    """Plots the timeseries and its ACF and PACF using Plotly.

    Parameters
    ----------
    timeseries : numpy array or pandas series.
        The timeseries.
    lags : int or array_like
        An int or array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int. If not provided,
        lags=np.arange(len(corr)) is used.
    alpha : real number in (0, 1]
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett’s formula. If None, no confidence intervals are plotted.
    ci : bool
        Useful when plotting the residuals of a time series. Overlays to the
        time series the std centered around zero.
    title : string
        The plot title.
    xaxis_title : string
        The x axis title.
    yaxis_title : string
        The y axis title.

    Returns
    -------
    fig : plotly figure
        The figure containing the three plots.

    """
    # create subplots
    fig = plotly.subplots.make_subplots(rows=3, vertical_spacing=0.1,
                                        subplot_titles=['Data', 'ACF',
                                                        'PACF'])
    # create data figure
    if 'index' not in timeseries.__dir__():
        index = np.arange(timeseries.size)
        values = timeseries
        fig.add_trace(go.Scatter(x=index, y=values, name='Values'), row=1,
                      col=1)
    else:
        index = timeseries.index
        values = timeseries.values
        fig.add_trace(go.Scatter(x=index, y=values, name='Values'), row=1,
                      col=1)
    if ci:
        # create 0 centered confidence interval at α confidence level
        ci = scipy.stats.norm.interval(alpha, scale=np.std(values))
        low_ci, high_ci = ci
        low_ci = np.repeat(low_ci, index.size)
        high_ci = np.repeat(high_ci, index.size)
        comp_index = np.concatenate((index, index[::-1]))
        ci = np.concatenate((low_ci, high_ci))
        # build plot trace
        name = 'Confidence interval centered in 0 at the ' + \
               '{}% confidence level'.format(100*(1-alpha))
        fig.add_trace(go.Scatter(x=comp_index, y=ci, name=name, fill='toself',
                                 opacity=0.5), row=1,
                      col=1)

    # create acf subplot
    acf, ci = stattools.acf(values, nlags=lags, fft=True, alpha=alpha)
    index = np.arange(acf.size)[1:]  # remove 0 lag
    acf = acf[1:]
    ci = ci[1:, :]
    # center to 0
    ci[:, 0] -= acf
    ci[:, 1] -= acf
    fig.add_trace(go.Scatter(x=index, y=acf, mode='markers', name='ACF'),
                  row=2, col=1)
    comp_index = np.concatenate((index, index[::-1]))
    ci_y = np.concatenate((ci[::, 0], ci[::-1, 1]))
    name = 'ACF Confidence interval at {}%'.format(100*(1-alpha))
    fig.add_trace(go.Scatter(x=comp_index, y=ci_y, fill='toself', opacity=0.5,
                             name=name), row=2, col=1)

    # create pacf subplot
    pacf, ci = stattools.pacf(values, nlags=lags, alpha=alpha)
    pacf = pacf[1:]  # remove 0 lag
    ci = ci[1:, :]
    # center to 0
    ci[:, 0] -= pacf
    ci[:, 1] -= pacf
    fig.add_trace(go.Scatter(x=index, y=pacf, mode='markers', name='PACF'),
                  row=3, col=1)
    ci_y = np.concatenate((ci[::, 0], ci[::-1, 1]))
    name = 'P' + name
    fig.add_trace(go.Scatter(x=comp_index, y=ci_y, fill='toself', opacity=0.5,
                             name=name), row=3, col=1)
    fig.update_layout(height=1100)

    # final layout changes
    fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
    fig.update_xaxes(title=xaxis_title, row=1, col=1)
    fig.update_xaxes(title='Lag', row=2, col=1)
    fig.update_xaxes(title='Lag', row=3, col=1)
    fig.update_yaxes(title=yaxis_title)
    fig.update_layout({'xaxis3.matches': 'x2'})
    fig.update_layout({'yaxis3.matches': 'y2'})
    return fig


def make_and_plot_predictions(fitted_mod, start_index=0, end_index=None,
                              alpha=0.95, title='', xaxis_title='',
                              yaxis_title='', pred_kwargs=None):
    """Plots one step ahead predictions for an univariate time series model.

    Parameters
    ----------
    fitted_mod : time series fitted model
        Must include all the data, train and test. To do this, create a model
        on a subset, fit it, and then add test data without fitting
    start_index : natural number
        Zero indexed position at which to start the one step ahead predictions.
    end_index : int
        Optional. Zero indexed position at which to end the one step ahead
        predictions.
    alpha : real number in [0, 1)
        The probability amplitude of the confidence interval.
    title : string
        The plot title.
    xaxis_title : string
        The x axis title.
    yaxis_title : string
        The y axis title.

    Returns
    -------
    fig : plotly Figure
        The plot.

    """
    # making predictions
    if pred_kwargs is not None:
        pred = fitted_mod.get_prediction(start=start_index, end=end_index,
                                         **pred_kwargs)
    else:
        pred = fitted_mod.get_prediction(start=start_index, end=end_index)
    pred_mean = pred.predicted_mean
    conf_int = pred.conf_int(alpha=alpha)
    # reconstruct data series
    data = pd.Series(index=fitted_mod.data.dates, data=fitted_mod.data.endog)
    # make figure
    fig = plot_already_made_predictions(data, pred_mean, conf_int, alpha,
                                        title, xaxis_title, yaxis_title)
    return fig


def plot_already_made_predictions(orig_data, pred_mean, conf_int=None,
                                  alpha=None, title='', xaxis_title='',
                                  yaxis_title=''):
    """Plots already made predictions against the original data.


    Parameters
    ----------
    orig_data : pandas Series
        The original data series.
    pred_mean : pandas Series
        The predicted mean
    conf_int : pandas DataFrame
        A DataFrame with two columns, containing the lower and upper bound of
        the each confidence interval respectively.
    alpha : real number in (0, 1]
        The alpha used to compute conf_int. Will be used only for the conf_int
        trace name.
    title : string
        The plot title.
    xaxis_title : string
        The x axis title.
    yaxis_title : string
        The y axis title.

    Returns
    -------
    fig : plotly figure
        The resulting figure.

    """
    fig = go.Figure()
    # displaying the test data
    fig.add_trace(go.Scatter(x=orig_data.index, y=orig_data.values,
                             name='Data'))
    # displaying the predictions
    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values,
                             name='Estimates', line={'color': '#E83D2E'}))
    # displaying the confidence interval and building the necessary objects
    if conf_int is not None:
        conf_int_shape = pd.concat((conf_int.iloc[:, 0],
                                    conf_int.iloc[::-1, 1]))
        name = 'Confidence interval at {}%'.format(100*alpha)
        fig.add_trace(go.Scatter(x=conf_int_shape.index,
                                 y=conf_int_shape.values,
                                 fill='toself', name=name, opacity=0.5,
                                 line={'color': '#E83D2E'}))

    # final layout changes
    fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(title=yaxis_title)
    return fig
