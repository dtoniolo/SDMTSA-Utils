from sklearn.model_selection import train_test_split


def ts_train_val_split(data, inputs, targets, train_size, val_size,
                       input_length):
    """Makes train and validation split for CV with time series data. The data
    must have already been reshaped for use with the NN.

    Parameters
    ----------
    data : pd DataFrame
        The data frame containing the original, flat data.
    inputs : indexable
        The input array for the NN.
    targets : indexable.
        The targets for the NN. Must have the same length as `inputs`.
    train_size : positive natural number or real number in (0, 1)
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the train split. If int, represents the
        absolute number of train samples
    val_size : positive natural number
        The number of observations to include in the validation set.
    input_length : positive natural number
        The number of timesteps taken as input by the neural network.

    Returns
    -------
    nn_data : iterable
        Contains:
            train_inputs : indexable
                The training inputs. They are of the same type as `inputs`.
            train_targets : indexable
                The training targets. They are of the same type as `targets`.
            val_inputs : indexable
                The validation inputs. They are of the same type as `inputs`.
            val_targets : indexable
                The validation targets. They are of the same type as `targets`.
    flat_data : iterable
        Contains:
            train_data : Pandas DataFrame
                The portion of the original, flat data used as input during
                training.
            val_data : Pandas DataFrame
                The portion of the original, flat data used  as
                a target for validation (plus the last target of the training
                set).
            train_pred_index : Pandas Index
                The index corresponding to the input predictions.
            val_pred_index : Pandas Index
                The index corresponding to the validation predictions.

    """
    subsets = train_test_split(inputs, targets, train_size=train_size,
                               shuffle=False)
    train_inputs, val_inputs, train_targets, val_targets = subsets
    if val_size > val_inputs.shape[0]:
        raise ValueError('Error during train and validation splitting for CV' +
                         'The requested size for the validation set is ' +
                         'greater than the non-training portion of the data.' +
                         ' To solve, lower `train_size` or `val_size`.')
    val_inputs = val_inputs[:val_size, ...]
    val_targets = val_targets[:val_size, ...]
    nn_data = train_inputs, val_inputs, train_targets, val_targets

    # separate train and val data and dates
    split_idx = len(train_inputs) + input_length - 1
    train_data = data.iloc[:split_idx, 0]
    val_data = data.iloc[split_idx:(split_idx+val_size), 0]
    train_pred_index = data.index[input_length:(split_idx+1)]
    val_pred_index = data.index[(split_idx+1):(split_idx+1+val_size)]
    flat_data = train_data, val_data, train_pred_index, val_pred_index
    return nn_data, flat_data
