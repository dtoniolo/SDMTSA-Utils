import pandas as pd
from .hour_changes import fix_hour_changes


def convert_hour(hour):
    hour = int(hour) - 1
    hour = str(hour).zfill(2)
    return hour


def custom_date_parser(dates, hours):
    for i, hour in enumerate(hours):
        hours[i] = convert_hour(hour)
    res = dates + ' ' + hours
    res = pd.to_datetime(res, format='%Y-%m-%d %H')
    return res


def load_data(path):
    data = pd.read_csv(path, sep=';', parse_dates=[['DATA', 'Ora']],
                       date_parser=custom_date_parser)
    data.set_index('DATA_Ora', inplace=True)
    # fix solar to legal hour changes
    fix_hour_changes(data)
    # fix missing data on 31 May 2020
    fill_data = (data['2020-05-24'].values + data['2020-06-7'].values) / 2
    fill_index = pd.date_range('2020-05-31 00:00:00', '2020-05-31 23:00:00',
                               freq='H')
    new_data = pd.DataFrame(index=fill_index, data=fill_data,
                            columns=['VALORE'])
    data = pd.concat((data, new_data), axis=0)
    data.sort_index(inplace=True)
    # check that everything is correct
    error_msg = "THe data doesn't have an exact hourly frequency"
    assert data.index.inferred_freq == 'H', error_msg
    # set data frequency
    data.index.freq = data.index.inferred_freq
    return data
