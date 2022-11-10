import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric


class OrigDataMAE(Metric):
    """A class used to compute the MAE on the original, non-scaled data.

    Parameters
    ----------
    scaler : sklearn scaler
        The object used to scale the data.
    name : str
        The name of the metric.
    **kwargs : type
        Keywords args for tf.keras.metrics.Metric.

    """
    def __init__(self, scaler, dtype=tf.keras.backend.floatx(),
                 name='original_data_mae', **kwargs):
        Metric.__init__(self, name=name, **kwargs)
        self.scale = tf.convert_to_tensor(scaler.scale_, dtype)
        self.my_dtype = dtype
        print('My dtype:', dtype)
        if hasattr(self, 'dtype'):
            print('The previous existing dtype is:', self.dtype)
        else:
            print('No dtype attribute.')
        self.n_samples = tf.zeros((1,), dtype)
        self.cum_absolute_error = self.add_weight(name='absolute_error',
                                                  initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert to numpy and create a new array to avoid modifying the tf
        # data
        print('UPDATING:')
        batch_size = tf.ones_like(y_true[:, 0])
        batch_size = tf.math.reduce_sum(batch_size, axis=0)
        self.n_samples += batch_size
        print('In input:')
        print('\ty_true.shape:', y_true.shape)
        print('\ty_pred.shape:', y_pred.shape)
        y_true = y_true / self.scale
        y_pred = y_pred / self.scale
        print('After scaling:')
        print('\ty_true.shape:', y_true.shape)
        print('\ty_pred.shape:', y_pred.shape)
        absolute_error = tf.abs(y_true - y_pred)
        print('Absolute difference:', absolute_error.shape)
        absolute_error = tf.math.reduce_sum(absolute_error, axis=(0, 1))
        print('Absolute error sum shape:', absolute_error.shape)
        absolute_error = batch_size * absolute_error
        self.cum_absolute_error.assign_add(absolute_error)
        print('AE:', absolute_error)
        print('AES:', self.cum_absolute_error)

    def result(self):
        print('AGGREGATING:')
        print('n_samples tensor shape:', self.n_samples.shape)
        self.cum_absolute_error = self.cum_absolute_error / self.n_samples
        print('Result MAE:', self.cum_absolute_error)
        return self.cum_absolute_error

    def reset_states(self):
        print('RESETTING:')
        self.n_samples = tf.zeros((1,), self.my_dtype)
        print('n_samples after reset:', self.n_samples)
        self.cum_absolute_error.assign(0.)
        print('absolute_errors after reset:', self.cum_absolute_error)


class OrigDataMAPE(Metric):
    """A class used to compute the MAPE on the original, non-scaled data.

    Parameters
    ----------
    scaler : sklearn scaler
        The object used to scale the data.
    name : str
        The name of the metric.
    **kwargs : type
        Keywords args for tf.keras.metrics.Metric.

    """
    def __init__(self, scaler, name='original_data_mape', **kwargs):
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
        ape = 100 * np.abs(y_true - y_pred) / y_true
        ape = ape.sum(axis=1)
        if hasattr(self, 'apes'):
            self.apes = np.concat([self.apes, ape])
        else:
            self.aes = ape

    def result(self):
        mae = self.apes.mean()
        return mae

    def reset_states(self):
        del self.apes
