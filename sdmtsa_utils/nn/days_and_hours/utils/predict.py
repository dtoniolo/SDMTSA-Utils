import numpy as np
import pandas as pd


def make_in_sample_predictions(model, input_data, output_data_index, scaler):
    """Makes in sample predictions.

    Parameters
    ----------
    model : tf model
        The model to use for the predictions.
    input_data : np or tf array
        Must be ready for it to be used by `model`.
    output_data_index : np array or Pandas Index
        The index to use when building the Pandas Series of the predictions.
    scaler : sklearn scaler
        Used to bring the data back to their original range.

    Returns
    -------
    pred : Pandas Series
        The time seris containing the predictions.

    """
    pred = model(input_data)
    # rescale predictions
    pred = pred.numpy().flatten()
    pred = scaler.inverse_transform(pred.reshape(-1, 1))
    pred = pred.flatten()
    # wrap them in a Pandas Series
    pred = pd.Series(data=pred, index=output_data_index)
    return pred


def make_oos_predictions(model, train_inputs, first_pred_timesamp, n_pred,
                         scaler):
    """Generates recurrent out of sample predictions for `model`.

    Parameters
    ----------
    model : tf Model
        A suitable neural network.
    train_inputs : array
        The array of inputs used in the training of the neural network.
    first_pred_timesamp : Pandas Timestamp
        The timestamp of the first prediction.
    n_pred : positive natural number.
        The number of predictions to do.
    scaler : sklearn scaler
        Used to bring the data back to their original range.

    Returns
    -------
    pred : Pandas Series
        The time seris containing the predictions.

    Notes
    -----
    Currently compatible only with networks that have output_length=1.

    """
    # generate the days and hours input data
    input_length = train_inputs.shape[1]
    start = first_pred_timesamp - input_length*pd.Timedelta(1, 'H')
    pred_index = pd.date_range(start, periods=n_pred+input_length, freq='H')
    days = pred_index.dayofweek
    hours = pred_index.hour
    pred_placeholder = np.empty(shape=days.shape, dtype=train_inputs.dtype)
    pred_placeholder[:input_length] = train_inputs[-1, :, 0]
    if train_inputs.shape[2] == 4:
        months = pred_index.month - 1
        data = np.stack((pred_placeholder, days, hours, months), axis=-1)
    else:
        data = np.stack((pred_placeholder, days, hours), axis=-1)

    for time_idx in range(input_length-1, len(data)-1):
        # create input for network
        input = np.empty((1, input_length, data.shape[1]), dtype=data.dtype)
        for feature_idx in range(data.shape[1]):
            if time_idx == input_length - 1:
                input[0, :, feature_idx] = data[time_idx::-1, feature_idx]
            else:
                slicer = slice(time_idx, time_idx-input_length, -1)
                input[0, :, feature_idx] = data[slicer, feature_idx]
        # make prediction
        pred = model(input)
        pred = pred.numpy()[0, 0]
        # update data
        data[time_idx+1, 0] = pred
    # extract the predictions and rescale them in the correct range
    pred = data[input_length:, 0].reshape(-1, 1)
    pred = scaler.inverse_transform(pred)
    # return predictions Series
    pred = pd.Series(data=pred.flatten(), index=pred_index[input_length:])
    return pred
