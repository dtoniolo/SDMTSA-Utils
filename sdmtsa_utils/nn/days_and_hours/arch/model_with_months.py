import tensorflow as tf
from tensorflow.keras.layers import Embedding
from . import cores


class ModelwEmbeddings2(tf.keras.Model):
    """Model that takes in input the time series value, the hour of the day,
    the day of the week and the month of the year.

    Parameters
    ----------
    input_length : positive natural number
        The number of timesteps to use as input for the model.
    output_length : positive natural number
        The number of timesteps the model will predict.
    core : str
        The name of the core. Currently the only valid one is 'M'.
    days_embedding_dim : positive natural number
        The size of the embeddings for the day of the week.
    hours_embedding_dim : positive natural number
        The size of the embeddings for the hour of the day.
    months_embedding_dim : positive natural number
        The size of the embeddings for the month.

    Attributes
    ----------
    days_emb_layer : tf Embedding layer
        The layer that generates the embeddings of the day of the week
        input variable.
    hours_emb_layer : tf Embedding layer
        The layer that generates the embeddings of the day of the week
        input variable.
    months_emb_layer : tf Embedding layer
        The layer that generates the embeddings of the month input variable.
    core : tf model
        The loader core.

    """
    def __init__(self, input_length, output_length, core, days_embedding_dim,
                 hours_embedding_dim, months_embedding_dim, **gru_kwargs):
        tf.keras.Model.__init__(self)
        self.days_emb_layer = Embedding(7, days_embedding_dim,
                                        input_length=input_length)
        self.hours_emb_layer = Embedding(24, hours_embedding_dim,
                                         input_length=input_length)
        self.months_emb_layer = Embedding(12, months_embedding_dim,
                                          input_length=input_length)

        core = self.get_core(core)
        core_input_shape = (input_length, 1 + days_embedding_dim +
                            hours_embedding_dim + months_embedding_dim)
        self.core = core(core_input_shape, output_length, **gru_kwargs)

    def get_core(self, core):
        """Loads a core from those available.

        Parameters
        ----------
        core : str
            The name of the core. Currently the only valid one is 'M'.

        Returns
        -------
        core : tf model
            The requested core.

        """
        if not hasattr(cores, core):
            raise ValueError('ModelwEmbeddings2 got an invalid name for ' +
                             ' the core architecture.')
        else:
            core = getattr(cores, core)
            return core

    def call(self, inputs, training=False):
        days_emb = self.days_emb_layer(inputs[:, :, 1])
        hours_emb = self.hours_emb_layer(inputs[:, :, 2])
        months_emb = self.months_emb_layer(inputs[:, :, 3])
        x = tf.concat([inputs[:, :, 0, None], days_emb, hours_emb, months_emb],
                      axis=2)
        x = self.core(x)
        return x
