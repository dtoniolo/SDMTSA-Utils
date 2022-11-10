def adfuller_wrapper(adfuller_result):
    """Pretty prints the result of an adfuller test

    Parameters
    ----------
    adfuller_result : statsmodels adfuller test result
        The result of an Augmented Dickey-Fuller test made by statsmodels.

    """
    print('Test statistic value:', adfuller_result[0], sep='\t')
    print('Estimated p-value:', adfuller_result[1], sep='\t')
    print('# of lags used:', adfuller_result[2], sep='\t')
    print('# of points:', adfuller_result[3], sep='\t')
    print('Level\tCritical value')
    for cv in adfuller_result[4]:
        print(cv, adfuller_result[4][cv], sep='\t')
