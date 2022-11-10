import tensorflow as tf
from tensorflow.keras.layers import Embedding
from . import cores


class ModelwEmbeddings(tf.keras.Model):
    """Model that takes in input the time series value, the hour of the day
    and the day of the week.

    Parameters
    ----------
    input_length : positive natural number
        The number of timesteps to use as input for the model.
    output_length : positive natural number
        The number of timesteps the model will predict.
    days_embedding_dim : positive natural number
        The size of the embeddings for the day of the week.
    hours_embedding_dim : positive natural number
        The size of the embeddings for the hour of the day.
    core : str
        The name of the core. Currently the only valid one is 'M'.

    Attributes
    ----------
    days_emb_layer : tf Embedding layer
        The layer that generates the embeddings of the day of the week
        input variable.
    hours_emb_layer : tf Embedding layer
        The layer that generates the embeddings of the day of the week
        input variable.
    core : tf model
        The loader core.

    """
    def __init__(self, input_length, output_length, days_embedding_dim,
                 hours_embedding_dim, core, **gru_kwargs):
        tf.keras.Model.__init__(self)
        self.days_emb_layer = Embedding(7, days_embedding_dim,
                                        input_length=input_length)
        self.hours_emb_layer = Embedding(24, hours_embedding_dim,
                                         input_length=input_length)

        core = self.get_core(core)
        core_input_shape = (input_length, 1 + days_embedding_dim +
                            hours_embedding_dim)
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
            raise ValueError('ModelwEmbeddings got an invalid name for ' +
                             ' the core architecture.')
        else:
            core = getattr(cores, core)
            return core

    def call(self, inputs, training=False):
        days_emb = self.days_emb_layer(inputs[:, :, 1])
        hours_emb = self.hours_emb_layer(inputs[:, :, 2])
        x = tf.concat([inputs[:, :, 0, None], days_emb, hours_emb], axis=2)
        x = self.core(x)
        return x
