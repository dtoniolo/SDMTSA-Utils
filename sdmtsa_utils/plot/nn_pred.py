import plotly.graph_objects as go


def plot_predictions(train_data, val_data, in_sample_pred, oos_pred,
                     oos_rec_pred, split=None):
    fig = go.Figure(go.Scatter(x=train_data.index, y=train_data.values,
                               name='Train Set'))
    fig.add_trace(go.Scatter(x=val_data.index, y=val_data.values,
                             name='Test Set'))
    fig.add_trace(go.Scatter(x=in_sample_pred.index, y=in_sample_pred.values,
                             name='In-Sample Predictions', mode='lines'))
    fig.add_trace(go.Scatter(x=oos_pred.index, y=oos_pred.values,
                             name='Validation Set One-Step-Ahead Predictions',
                             mode='lines'))
    fig.add_trace(go.Scatter(x=oos_rec_pred.index, y=oos_rec_pred.values,
                             name='Out of Sample Recurrent Predictions',
                             mode='lines'))
    # final layout changes
    title = 'Train and Val Set Predictions '
    if split is not None:
        title += '(Train Set {}% of Total)'.format(100*split)
    fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'},
                      xaxis_title='Date and Time', yaxis_title='Value')
    return fig
