import tensorflow as tf


class OrigDataMAE():
    """A class used to compute the MAE on the original, non-scaled data.

    Parameters
    ----------
    scaler : sklearn scaler
        The object used to scale the data.
    name : str
        The name of the metric.

    """
    def __init__(self, scaler, dtype=tf.keras.backend.floatx(),
                 name='original_data_mae'):
        self.scale = tf.convert_to_tensor(scaler.scale_, dtype)
        self.name = name

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = y_true / self.scale
        y_pred = y_pred / self.scale
        ae = tf.abs(y_true - y_pred)
        ae = tf.math.reduce_sum(ae, axis=1)
        mae = tf.math.reduce_mean(ae, axis=0)
        return mae


class OrigDataMAPE():
    """A class used to compute the MAPE on the original, non-scaled data.

    Parameters
    ----------
    scaler : sklearn scaler
        The object used to scale the data.
    name : str
        The name of the metric.

    """
    def __init__(self, scaler, dtype=tf.keras.backend.floatx(),
                 name='original_data_mape'):
        self.scale = tf.convert_to_tensor(scaler.scale_, dtype)
        self.offset = tf.convert_to_tensor(scaler.data_min_, dtype)
        self.name = name

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = y_true / self.scale + self.offset
        y_pred = y_pred / self.scale + self.offset
        ape = 100 * tf.abs(y_true - y_pred) / y_true
        ape = tf.math.reduce_sum(ape, axis=1)
        mape = tf.math.reduce_mean(ape, axis=0)
        return mape
