from typing import Any, Literal, Union, Optional
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import plotly
import plotly.subplots
import plotly.graph_objects as go
from statsmodels.tsa import stattools
import scipy
import scipy.stats


def plot_ts(
    timeseries: Union[np.ndarray, pd.Series],
    lags: Union[int, ArrayLike],
    alpha: float = 0.05,
    plot_ci: bool = False,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    missing: Union[Literal["raise"], Literal["conservative"]] = "raise",
) -> go.Figure:
    """Plots the timeseries and its ACF and PACF using Plotly.

    Parameters
    ----------
    timeseries : numpy array or pandas series.
        The timeseries.
    lags : int or array_like
        An int or array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int. If not provided,
        lags=np.arange(len(corr)) is used.
    alpha : real number in the interval :math:`(0, 1]`
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett’s formula. If None, no confidence intervals are plotted.
    plot_ci : bool
        Useful when plotting the residuals of a time series. Overlays to the time
        series the std centered around zero.
    title : string
        The plot title.
    xaxis_title : string
        The x axis title.
    yaxis_title : string
        The y axis title.
    missing : "raise" or "conservative"
        If ``"raise"`` then an exception will be risen if NaN values are found,
        otherwise ``"conservative"`` computes the autocovariance using nan-ops so that
        nans are removed when computing the mean and cross-products that are used to
        estimate the autocovariance. In the latter case, if NaN values are present the
        PACF won't be plotted.

    Returns
    -------
    fig : plotly figure
        The figure containing the three plots.
    
    Notes
    -----
    If `timeseries` is a Pandas series, then its index will be used for plotting.
    Otherwise, a simple incremental index ranging from 1 to ``len(timeseries)``
    (included) will be created.

    """
    # create subplots
    if missing not in ["raise", "conservative"]:
        raise TypeError(
            "`missing` but be either 'raise' or 'conservative', got '{missing}'."
        )
    if isinstance(timeseries, np.ndarray):
        has_nans = np.any(np.isnan(timeseries))
    else:
        has_nans = timeseries.isna().any()
    if has_nans:
        if missing == "raise":
            raise ValueError(
                "Found missing values in the input data, but there should be none."
            )
        plot_pacf = False
        rows = 2
    else:
        plot_pacf = True
        rows = 3

    fig = plotly.subplots.make_subplots(
        rows=rows, vertical_spacing=0.1, subplot_titles=["Data", "ACF", "PACF"]
    )
    # create data figure
    if isinstance(timeseries, pd.Series):
        index = timeseries.index
        values = timeseries.values
        fig.add_trace(go.Scatter(x=index, y=values, name="Values"), row=1, col=1)
    else:
        # make the index of the timeseries start from 1
        index = np.arange(1, timeseries.size + 1)
        values = timeseries
        fig.add_trace(go.Scatter(x=index, y=values, name="Values"), row=1, col=1)
    if plot_ci:
        # create 0 centered confidence interval at α confidence level
        ci = scipy.stats.norm.interval(alpha, scale=np.std(values))
        low_ci, high_ci = ci
        low_ci = np.repeat(low_ci, index.size)
        high_ci = np.repeat(high_ci, index.size)
        comp_index = np.concatenate((index, index[::-1]))
        ci = np.concatenate((low_ci, high_ci))
        # build plot trace
        name = (
            "Confidence interval centered in 0 at the "
            + "{}% confidence level".format(100 * (1 - alpha))
        )
        fig.add_trace(
            go.Scatter(x=comp_index, y=ci, name=name, fill="toself", opacity=0.5),
            row=1,
            col=1,
        )

    # create acf subplot
    acf, ci = stattools.acf(values, nlags=lags, fft=True, alpha=alpha, missing=missing)
    index = np.arange(acf.size)[1:]  # remove 0 lag
    acf = acf[1:]
    ci = ci[1:, :]
    # center to 0
    ci[:, 0] -= acf
    ci[:, 1] -= acf
    fig.add_trace(go.Scatter(x=index, y=acf, mode="markers", name="ACF"), row=2, col=1)
    comp_index = np.concatenate((index, index[::-1]))
    ci_y = np.concatenate((ci[::, 0], ci[::-1, 1]))
    name = "ACF Confidence interval at {}%".format(100 * (1 - alpha))
    fig.add_trace(
        go.Scatter(x=comp_index, y=ci_y, fill="toself", opacity=0.5, name=name),
        row=2,
        col=1,
    )

    if plot_pacf:
        # create pacf subplot
        pacf, ci = stattools.pacf(values, nlags=lags, alpha=alpha)
        pacf = pacf[1:]  # remove 0 lag
        ci = ci[1:, :]
        # center to 0
        ci[:, 0] -= pacf
        ci[:, 1] -= pacf
        fig.add_trace(
            go.Scatter(x=index, y=pacf, mode="markers", name="PACF"), row=3, col=1
        )
        ci_y = np.concatenate((ci[::, 0], ci[::-1, 1]))
        name = "P" + name
        fig.add_trace(
            go.Scatter(x=comp_index, y=ci_y, fill="toself", opacity=0.5, name=name),
            row=3,
            col=1,
        )
        fig.update_layout(height=1100)
        fig.update_layout({"xaxis3.matches": "x2"})
        fig.update_layout({"yaxis3.matches": "y2"})
    else:
        fig.update_layout(height=800)

    # final layout changes
    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"})
    fig.update_xaxes(title=xaxis_title, row=1, col=1)
    fig.update_xaxes(title="Lag", row=2, col=1)
    fig.update_xaxes(title="Lag", row=3, col=1)
    fig.update_yaxes(title=yaxis_title)
    return fig


def make_and_plot_predictions(
    fitted_mod,
    start_index: int = 0,
    end_index: Optional[int] = None,
    alpha: float = 0.95,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    pred_kwargs: Optional[Any] = None,
) -> go.Figure:
    """Plots one step ahead predictions for an univariate time series model.

    Parameters
    ----------
    fitted_mod : time series fitted model
        Must include all the data, train and test. To do this, create a model
        on a subset, fit it, and then add test data without fitting
    start_index : natural number
        Zero indexed position at which to start the one step ahead predictions.
    end_index : int, optional
        Optional. Zero indexed position at which to end the one step ahead
        predictions.
    alpha : real number in [0, 1)
        The probability amplitude of the confidence interval.
    title : str
        The plot title.
    xaxis_title : str
        The x axis title.
    yaxis_title : str
        The y axis title.
    pred_kwargs : Any, optional
        If given, they will be passed to the ``get_predictions()`` method of
        `fitted_mod`.

    Returns
    -------
    fig : plotly Figure
        The plot.

    """
    # making predictions
    if pred_kwargs is not None:
        pred = fitted_mod.get_prediction(
            start=start_index, end=end_index, **pred_kwargs
        )
    else:
        pred = fitted_mod.get_prediction(start=start_index, end=end_index)
    pred_mean = pred.predicted_mean
    conf_int = pred.conf_int(alpha=alpha)
    # reconstruct data series
    if fitted_mod.data.dates is None:
        if end_index is None:
            end_index = len(fitted_mod.data.endog)
        # plot index starting from 1
        index = range(start_index + 1, end_index + 1)
        pred_mean = pd.Series(index=index, data=pred_mean)
        conf_int = pd.DataFrame(index=index, data=conf_int)
        data = pd.Series(
            index=range(1, len(fitted_mod.data.endog) + 1), data=fitted_mod.data.endog
        )
    else:
        data = pd.Series(index=fitted_mod.data.dates, data=fitted_mod.data.endog)
    # make figure
    fig = plot_already_made_predictions(
        data, pred_mean, conf_int, alpha, title, xaxis_title, yaxis_title
    )
    return fig


def plot_already_made_predictions(
    orig_data: pd.Series,
    pred_mean: pd.Series,
    conf_int: Optional[pd.DataFrame] = None,
    alpha: Optional[float] = None,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    """Plots already made predictions against the original data.


    Parameters
    ----------
    orig_data : pandas Series
        The original data series.
    pred_mean : pandas Series
        The predicted mean
    conf_int : pandas DataFrame, optional
        If given, it must be a DataFrame with two columns, containing the lower and
        upper bound of the confidence interval of each point.
    alpha : real number in (0, 1], optional
        The alpha that was used to compute ``conf_int``. Will be used only for the
        name of the trace that shows the confidence interval.
    title : str
        The title of the plot.
    xaxis_title : str
        The title of the x axis.
    yaxis_title : str
        The title of the x axis.

    Returns
    -------
    fig : plotly figure
        The resulting figure.

    """
    if alpha is not None:
        if conf_int is None:
            raise ValueError(
                "Received a value for α, but didn't receive the bounds of the "
                "confidence interval."
            )
        if alpha <= 0 or alpha > 1:
            raise ValueError("α must belong to the (0, 1] interval.")
    if conf_int is not None:
        if len(conf_int.columns) != 2:
            raise ValueError(
                "The data frame storing the bounds for the confidence intervals "
                f"must have two columns, got {len(conf_int.columns)}."
            )
        if (conf_int.iloc[:, 0] > conf_int.iloc[:, 1]).any():
            raise ValueError(
                "The lower bound of each confidence interval must be equal or lower "
                "to the corresponding upper bound."
            )
        if len(pred_mean) != len(conf_int):
            raise ValueError(
                "Received a diffent number of predictions and confidence intervals."
            )
        if (pred_mean.index != conf_int.index).any():
            raise ValueError(
                "The predictions series and the confidence intervals data frame must "
                "have the same index."
            )
        if (conf_int.iloc[:, 0] > pred_mean).any() or (
            pred_mean > conf_int.iloc[:, 1]
        ).any():
            raise ValueError(
                "Each prediction must lie inside the corresponding confidence "
                "interval."
            )

    fig = go.Figure()
    # displaying the test data
    fig.add_trace(go.Scatter(x=orig_data.index, y=orig_data.values, name="Data"))
    # displaying the predictions
    fig.add_trace(
        go.Scatter(
            x=pred_mean.index,
            y=pred_mean.values,
            name="Estimates",
            line={"color": "#E83D2E"},
        )
    )
    # displaying the confidence interval and building the necessary objects
    if conf_int is not None:
        conf_int_shape = pd.concat((conf_int.iloc[:, 0], conf_int.iloc[::-1, 1]))
        name = "Confidence interval at {}%"
        if alpha is not None:
            name = name.format(100 * alpha)
        else:
            name = name.format("?")
        fig.add_trace(
            go.Scatter(
                x=conf_int_shape.index,
                y=conf_int_shape.values,
                fill="toself",
                name=name,
                opacity=0.5,
                line={"color": "#E83D2E"},
            )
        )

    # final layout changes
    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"})
    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(title=yaxis_title)
    return fig
