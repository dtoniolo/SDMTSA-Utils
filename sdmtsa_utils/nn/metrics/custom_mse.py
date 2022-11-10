import tensorflow as tf


class OrigDataMSE():
    """A class used to compute the MSE on the original, non-scaled data.

    Parameters
    ----------
    scaler : sklearn scaler
        The object used to scale the data.
    name : str
        The name of the metric.

    """
    def __init__(self, scaler, dtype=tf.keras.backend.floatx(),
                 name='original_data_mse'):
        self.scale = tf.convert_to_tensor(scaler.scale_, dtype)
        self.name = name

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = y_true / self.scale
        y_pred = y_pred / self.scale
        ae = (y_true - y_pred)**2
        ae = tf.math.reduce_sum(ae, axis=1)
        mae = tf.math.reduce_mean(ae, axis=0)
        return mae
