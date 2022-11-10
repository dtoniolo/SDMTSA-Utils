import numpy as np
import pandas as pd


def undiff_estimates(last_est, origs, ks, conf_int=None):
    # check input args
    start_pred, end_sample, end_pred = _check_undiff_est_args(last_est, origs,
                                                              ks, conf_int)

    # execute
    estimates = list()

    for i in range(len(origs)-1, -1, -1):
        ki = ks[i]
        diff = origs[i]
        if i == len(origs)-1:
            prev_diff_est = last_est
        else:
            prev_diff_est = estimates[-1]
        diff_est = pd.Series(data=np.array(prev_diff_est.values),
                             index=last_est.index)
        if end_sample + ki <= end_pred:
            for idx in pd.date_range(start_pred, end_sample+ki):
                diff_est[idx] = prev_diff_est[idx] + diff[idx - ki]
            for idx in pd.date_range(end_sample+ki, end_pred, closed='right'):
                diff_est[idx] = prev_diff_est[idx] + diff_est[idx - ki]
        else:
            dr = pd.date_range(start_pred-ki, end_pred-ki)
            print(dr)
            for idx in pd.date_range(start_pred, end_pred):
                diff_est[idx] = prev_diff_est[idx] + diff[idx - ki]
        estimates.append(diff_est)
    estimates = estimates[::-1]

    if conf_int is not None:
        conf_int_lower = conf_int.iloc[:, 0]
        conf_int_lower, _ = undiff_estimates(conf_int_lower, origs, ks)
        conf_int_upper = conf_int.iloc[:, 1]
        conf_int_upper, _ = undiff_estimates(conf_int_upper, origs, ks)
        undiff_conf_int = pd.DataFrame(conf_int)
        undiff_conf_int.iloc[:, 0] = conf_int_lower.values
        undiff_conf_int.iloc[:, 1] = conf_int_upper.values
    else:
        undiff_conf_int = None

    return estimates[0], undiff_conf_int


def _check_undiff_est_args(last_est, origs, ks, conf_int):
    # check inputs
    if len(origs) != len(ks):
        raise ValueError('Invalid input. Should have the same number of ' +
                         'series and of ks.')
    if not isinstance(last_est, pd.Series):
        raise ValueError('Invalid input. The estimate of the last series ' +
                         'should be a Pandas Series, but is ' +
                         '{} instad.'.format(type(last_est)))
    for i, series in enumerate(origs):
        if not isinstance(series, pd.Series):
            raise ValueError('Invalid input. Each data series should be a ' +
                             'Pandas Series, but the #{} is '.format(i+1) +
                             '{} instad.'.format(type(series)))
    for i, ki in enumerate(ks):
        if not isinstance(ki, pd.Timedelta):
            raise ValueError('Invalid input. Each ki series should be a ' +
                             'Pandas Timedelta, but the #{} is '.format(i+1) +
                             '{} instad.'.format(type(ki)))
    if conf_int is not None:
        if not isinstance(conf_int, pd.DataFrame):
            raise ValueError('Invalid input. The conf_int argument should be' +
                             ' a Pandas DataFrame, but is ' +
                             '{} instad.'.format(type(conf_int)))
        if conf_int.shape[1] != 2:
            raise ValueError('Invalid input. The conf_int argument should be' +
                             ' a data frame with two columns, but has ' +
                             '{} instead.'.format(conf_int.shape[1]))
        if conf_int.shape[0] != len(last_est):
            raise ValueError('Invalid input. The conf_int argument should ' +
                             'have the same number of rows as last_ord ' +
                             '({}), but has '.format(len(last_est)) +
                             '{} instead.'.format(conf_int.shape[0]))
        if np.any(conf_int.index != last_est.index):
            raise ValueError('Invalid input. The indexes of last_est and  ' +
                             'conf_int should be equal everywhere, but they ' +
                             'aren\'t.')

    starts = list()
    for i, series in enumerate(origs):
        start = series.index[0]
        end = series.index[-1]
        if i > 0:
            if start != starts[-1] + ks[i-1]:
                raise ValueError('Invalid input. Each start should be equal ' +
                                 'to the previous one plus ki, but the ' +
                                 '#{} is {} instead of '.format(i+1, start) +
                                 '{}.'.format(starts[-1] + ks[i-1]))
            if end != end_sample:
                raise ValueError('Invalid input. All ends should be equal, ' +
                                 'but the #{} is {} instead'.format(i+1, end) +
                                 'of {}.'.format(end_sample))
        starts.append(start)
        end_sample = end
    start_pred = last_est.index[0]
    end_pred = last_est.index[-1]
    if start_pred != starts[-1] + ks[-1]:
        raise ValueError('Invalid input. The estimates of the highest oreder' +
                         ' diff shoul start at ' +
                         '{}, but '.format(starts[-1] + ks[-1]) +
                         'begin at {} instead.'.format(start_pred))

    return start_pred, end_sample, end_pred
'''

def undiff_estimates2(last_est, origs, ks, start_pred_idx, end_pred_idx,
                      conf_int=None):
    # check input args
    _check_undiff_est_args2(last_est, origs,
                                                               ks,
                                                               start_pred_idx,
                                                               end_pred_idx,
                                                               conf_int)

    # execute
    estimates = list()
    ks = np.array(ks)
    end_sample_idx = len(origs[0])
    delta = end_pred_idx - end_sample_idx
    for i in range(len(origs)-1, -1, -1):
        ki = ks[i]
        diff = origs[i]
        if i == len(origs)-1:
            kcumsumi = 0
            prev_diff_est = last_est
        else:
            kcumsumi = ks[(i+1):].sum()
            prev_diff_est = estimates[-1]
        diff_est = pd.Series(data=np.array(prev_diff_est.values),
                             index=last_est.index)
        if end_sample + ki <= end_pred:
            for idx in pd.date_range(start_pred, end_sample+ki):
                diff_est[idx] = prev_diff_est[idx] + diff[idx - ki]
            for idx in pd.date_range(end_sample+ki, end_pred, closed='right'):
                diff_est[idx] = prev_diff_est[idx] + diff_est[idx - ki]
        if delta <= 0:
            diff_est.values = prev_diff_est.values + diff[kcumsumi:(-ki+delta)]
        else:

        estimates.append(diff_est)
    estimates = estimates[::-1]

    if conf_int is not None:
        conf_int_lower = conf_int.iloc[:, 0]
        conf_int_lower, _ = undiff_estimates(conf_int_lower, origs, ks)
        conf_int_upper = conf_int.iloc[:, 1]
        conf_int_upper, _ = undiff_estimates(conf_int_upper, origs, ks)
        undiff_conf_int = pd.DataFrame(conf_int)
        undiff_conf_int.iloc[:, 0] = conf_int_lower.values
        undiff_conf_int.iloc[:, 1] = conf_int_upper.values
    else:
        undiff_conf_int = None

    return estimates[0], undiff_conf_int
'''

def _check_index(index):
    for i in range(1, len(index)):
        if not index[i] >= index[i+1]:
            raise ValueError('Invalid input.Each index must be non ' +
                             'decreasing.')


def _check_undiff_est_args2(last_est, origs, ks, start_pred_idx, end_pred_idx,
                            conf_int):
    # check inputs
    if len(origs) != len(ks):
        raise ValueError('Invalid input. Should have the same number of ' +
                         'series and of ks.')
    if not isinstance(last_est, pd.Series):
        raise ValueError('Invalid input. The estimate of the last series ' +
                         'should be a Pandas Series, but is ' +
                         '{} instad.'.format(type(last_est)))
    for i, series in enumerate(origs):
        if not isinstance(series, pd.Series):
            raise ValueError('Invalid input. Each data series should be a ' +
                             'Pandas Series, but the #{} is '.format(i+1) +
                             '{} instad.'.format(type(series)))
    for i, ki in enumerate(ks):
        if not isinstance(ki, int):
            raise ValueError('Invalid input. Each ki should be an integer, ' +
                             'but #{} is {} instead'.format(i+1, type(ki)))
        if ki <= 0:
            raise ValueError('Invalid input. Each ki should be positive, but' +
                             '#{} isn\'t'.format(i+1))
    if not isinstance(start_pred_idx, int):
        raise ValueError('Invalid input. start_pred_idx a should be an ' +
                         'integer.')
    if start_pred_idx <= 0:
        raise ValueError('Invalid input. start_pred_idx should be positive.')
    if not isinstance(end_pred_idx, int):
        raise ValueError('Invalid input. end_pred_idx a should be an ' +
                         'integer.')
    if end_pred_idx <= 0:
        raise ValueError('Invalid input. end_pred_idx should be positive.')
    if start_pred_idx > end_pred_idx:
        raise ValueError('Invalid input. end_pred_idx should be greater or ' +
                         'equal than start_pred_idx.')
    if conf_int is not None:
        if not isinstance(conf_int, pd.DataFrame):
            raise ValueError('Invalid input. The conf_int argument should be' +
                             ' a Pandas DataFrame, but is ' +
                             '{} instad.'.format(type(conf_int)))
        if conf_int.shape[1] != 2:
            raise ValueError('Invalid input. The conf_int argument should be' +
                             ' a data frame with two columns, but has ' +
                             '{} instead.'.format(conf_int.shape[1]))
        if conf_int.shape[0] != len(last_est):
            raise ValueError('Invalid input. The conf_int argument should ' +
                             'have the same number of rows as last_ord ' +
                             '({}), but has '.format(len(last_est)) +
                             '{} instead.'.format(conf_int.shape[0]))
        if np.any(conf_int.index != last_est.index):
            raise ValueError('Invalid input. The indexes of last_est and  ' +
                             'conf_int should be equal everywhere, but they ' +
                             'aren\'t.')
    if end_pred_idx > len(origs[0]):
        last_orig = len(origs[0])
    else:
        last_orig = end_pred_idx
    last_pred = last_orig - start_pred_idx
    if not np.all(last_est.index[:last_pred] ==
                  origs[0].index[start_pred_idx:last_orig]):
        raise ValueError('Invalid input. The estimates and original data ' +
                         'indexes do not coincide.')
    ks = np.array(ks)
    for i in range(1, len(origs)):
        ki = ks[i-1]
        if i == 1:
            kcumsumi = 0
        else:
            kcumsumi = ks[:(i-2)].cumsum()
        orig = origs[0]
        diff = origs[i]
        if not np.all(diff.index == orig.index[kcumsumi:]):
            raise ValueError('The index of differentiated series ' +
                             '#{} doesn\'t match the one of the '.format(i) +
                             'original series.')

    _check_index(last_est.index)
    for series in origs:
        _check_index(series.index)
    if conf_int is not None:
        _check_index(conf_int.index)
