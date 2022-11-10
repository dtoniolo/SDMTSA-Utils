def update_titles_(fig, title, xaxis_title, yaxis_title):
    """Adds to the figure the global, x axis and y axis title. The global title
    is centered.

    Used to avoid coping and pasting code every time.

    Parameters
    ----------
    fig : plotly Figure
        The figure to which add the titles.
    title : str
        The title of the figure.
    xaxis_title : str
        The title of the x axis.
    yaxis_title : str
        The title of the y axis.

    Notes
    -----
    The figure is modified in place.

    """
    # final layout changes
    fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(title=yaxis_title)
