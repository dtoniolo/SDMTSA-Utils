import numpy as np
from tensorflow.keras.metrics import Metric


class OrigDataMSE(Metric):
    """A class used to compute the MSE on the original, non-scaled data.

    Parameters
    ----------
    scaler : sklearn scaler
        The object used to scale the data.
    name : str
        The name of the metric.
    **kwargs : type
        Keywords args for tf.keras.metrics.Metric.

    """
    def __init__(self, scaler, name='original_data_mse', **kwargs):
        Metric.__init__(self, name=name, **kwargs)
        self.scaler = scaler

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert to numpy and create a new array to avoid modifying the tf
        # data
        if hasattr(y_true, 'numpy'):
            y_true = y_true.numpy()
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()
        y_true = self.scaler.inverse_transform(y_true)
        y_pred = self.scaler.inverse_transform(y_pred)
        se = (y_true - y_pred)**2
        se = se.sum(axis=1)
        if hasattr(self, 'ses'):
            self.ses = np.concat([self.ses, se])
        else:
            self.ses = se

    def result(self):
        mse = self.ses.mean()
        return mse

    def reset_states(self):
        del self.ses
