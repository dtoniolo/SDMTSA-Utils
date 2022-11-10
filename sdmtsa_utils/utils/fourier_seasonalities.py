import numpy as np


def create_fourier_seasonalities(timesteps, period, harmonics):
    """Creates a matrix of dimentions timesteps x 2*harmonics containing the
    cosine and sine values for each timestep and harmonic.

    Parameters
    ----------
    timesteps : positive natural number
        The number of timestep to produce.
    period : positive real number
        The fundamental period of the fourier seasonalities, in timesteps units.
    harmonics : positive natural number
        The number of harmonics to produce.

    Returns
    -------
    fourier season : numpy array
        Of shape (timesteps, 2*harmonics), containing the cosine and sine
        values for each timestep and harmonic.

    """
    fourier_season = np.empty(shape=(timesteps, 2*harmonics), dtype='float64')
    timesteps = np.arange(timesteps)
    for i in range(harmonics):
        cos = np.cos(2*np.pi * (i+1) * timesteps / period)
        sin = np.sin(2*np.pi * (i+1) * timesteps / period)
        fourier_season[:, 2*i] = cos
        fourier_season[:, 2*i+1] = sin
    return fourier_season
