import numpy as np
import pandas as pd

# first italian lockdown
first_lockdown = pd.date_range('2020-03-09', '2020-05-18', freq='H')


def get_covid_dummy(index):
    start_index = np.where(index == first_lockdown[0])[0][0]
    end_index = np.where(index == first_lockdown[-1])[0][0] + 1
    assert np.all(index[start_index:end_index] == first_lockdown)
    dummy_covid = np.zeros(len(index), dtype='bool')
    dummy_covid[start_index:end_index] = True
    return dummy_covid
