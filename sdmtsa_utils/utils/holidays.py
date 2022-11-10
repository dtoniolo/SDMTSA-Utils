import numpy as np
import holidays

italian_holidays = holidays.IT()


def get_holiday_binary_array(index):
    """Cretes a binary array indicating whether a given instant in time is an
    Italian holiday or not.

    Parameters
    ----------
    index : pandas DateTime Index
        The index containing the dates.

    Returns
    -------
    holidays_array : numpy array
        Of shape (len(index),) and binary dtype. Holds True if a given instant
        int time is an italian holiday.

    """
    holidays_array = np.zeros((len(index),), dtype='bool')
    for i, time in enumerate(index):
        if time in italian_holidays:
            holidays_array[i] = True
    return holidays_array
