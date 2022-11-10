import numpy as np
import pandas as pd


def fix_hour_changes(data):
    """Used to regularize a time series that has jumps because of the legal to
    solar and solar to legal time changes.

    Adds the missing value at 2:00 AM of the last Sunday of March by averaging
    the ones from the previous and the next hours and removes the duplicate
    hour at 3:00 AM in the last Sunday of October.



    Parameters
    ----------
    data : pandas DataFrame
        Indexed by an hourly time index.

    Note
    ----
    The data frame is modified in place.

    """
    fix_solar_to_legal(data)
    fix_legal_to_solar(data)


def fix_legal_to_solar(data):
    """Removes the duplicate hour at 3:00 AM in the last Sunday of October.

    Used to regularize a time series that has jumps because of the legal to
    solar time change.

    Parameters
    ----------
    data : pandas DataFrame
        Indexed by an hourly time index.

    Note
    ----
    The data frame is modified in place.

    """
    years = data.index.year.unique()
    # iterate over years
    for year in years:
        # retrieve the last sunday of march of this year
        group = data[data.index.year == year]
        october = group.index[group.index.month == 10]
        october_sundays = october[october.day_name() == 'Sunday']
        last_sunday = october_sundays[october_sundays.day ==
                                      october_sundays.day.max()]
        last_sunday_three_o_clock = last_sunday[last_sunday.hour == 3]
        # there may be no march, because the time series starts later or end
        # earlier than October
        if len(last_sunday_three_o_clock) != 0:
            if len(last_sunday_three_o_clock) == 1:
                continue
            if len(last_sunday_three_o_clock) != 2:
                raise ValueError('In year {} '.format(year) +
                                 '{}, but'.format(last_sunday_three_o_clock) +
                                 ' it should be of length 0, 1 or 2.')
            # create mask
            last_sunday_three_o_clock = last_sunday_three_o_clock[0]
            mask = data.index == last_sunday_three_o_clock
            mask[np.where(mask)[0]] = False
            mask = np.logical_not(mask)
            data = data.iloc[mask, :]


def fix_solar_to_legal(data):
    """Adds the missing value at 2:00 AM of the last Sunday of March by
    averaging the ones from the previous and the next hours.

    Used to regularize a time series that has jumps because of the solar to
    legal time change

    Parameters
    ----------
    data : pandas DataFrame
        Indexed by an hourly time index.

    Note
    ----
    The data frame is modified in place.

    """
    years = data.index.year.unique()
    # iterate over years
    for year in years:
        # retrieve the last sunday of march of this year
        group = data[data.index.year == year]
        march = group.index[group.index.month == 3]
        march_sundays = march[march.day_name() == 'Sunday']
        last_sunday = march_sundays[march_sundays.day ==
                                    march_sundays.day.max()]
        last_sunday_one_o_clock = last_sunday[last_sunday.hour == 1]
        # there may be no march, because the time series starts later or end
        # earlier than March
        if len(last_sunday_one_o_clock) != 0:
            last_sunday_one_o_clock = last_sunday_one_o_clock[0]
            one_hour = pd.Timedelta(1, unit='H')
            last_sunday_two_o_clock = last_sunday_one_o_clock + one_hour
            # copy data
            values = group.loc[last_sunday_one_o_clock]
            values2 = group.loc[last_sunday_one_o_clock + 2*one_hour]
            data.loc[last_sunday_two_o_clock] = (values + values2) / 2
    # sort the index
    data.sort_index(inplace=True)
