"""
Contains tools for timeseries analysis
"""

import pandas as pd
import numpy as np
import operator
import itertools

def causal_smooth(x, window_size, window_function='box', nan_start=False, trim_front=0):
    """ 
        Casually smooths a timeseries, where causal means that we don't use any future points to smooth the past.
        Nan elements are ignored.
        
    Parameters
    ----------
        x : array-like, shape (n,)
            
        window_size: int
            Size of the sliding window. Must be odd.
            
        window_function: str, either "box" or 'triangle'.
        
        nan_start: bool
            If true, begining of returned array will be nan for indexes < window_size
            (i.e. smoothing doesn't start until there is a full window)
            
        trim_front: int
            if trim_front is a value > 0, the first trim_front elements are set to nan
    
    Returns
    -------
        v_smoothed: array-like, shape (n,)

    """
    assert nan_start in {True, False}
    # causal means that we don't use any future points to smooth the past.
    x = np.squeeze(x)

    # Trivial cases in which we don't smooth
    if window_size == -1 or window_size == 0 or window_size is None: return x

    # Check input
    if type(x) is list: x = np.asarray(x)
    if x.ndim != 1: raise ValueError(f"Vector must be 1d. Vect shape =  {x.shape}")

    # Start with a full nan array
    v_smoothed = np.full(x.shape, fill_value=np.nan, dtype=np.float64)

    # Calculate what our weights will be
    if window_function in {'triangle', 'triangular', 'tri'} :
        n = np.arange(1, window_size + 1)
        N = len(n)
        w = 1 - (N - n) / N
    elif window_function == 'box':
        w = np.ones(shape=(window_size, ))
    else: raise ValueError()

    # Do we ignore the first window size of trials?
    if nan_start: start = window_size
    else: start = 0

    # Enumerate over our array
    for _, i in enumerate(np.arange(start, len(x))):
        window = x[ np.max( [i - window_size, 0] ): i] # Get our window

        # If we have a nan_start, this guarentees the size of our window is window_size. Else could be shorter
        if nan_start: assert len(window) == window_size
        assert len(window) <= window_size

        # Find out where we have nans, for later reference.
        is_finite = np.isfinite(window)

        # Cut our weights down to the size of our window.
        w_window = w[:len(window)]
        if nan_start or i > 50: assert np.all(w_window == w)

        # If we don't yet have any valid points, we set to a nan.
        if len(window[is_finite]) == 0:
            v_smoothed[i] = np.nan
        else:
            try:
                # Smoothed is elementwise multiplation of weights and values divided by
                v_smoothed[i] = np.nansum(w_window[is_finite] * window[is_finite]) / np.nansum(w_window[is_finite])
            except:
                print(w_window[is_finite])
                print(np.nansum(w_window[is_finite] * window[is_finite]))
                print(np.nansum(w_window[is_finite]))
                print(np.nansum(w_window[is_finite] * window[is_finite]) / np.nansum(w_window[is_finite]))
                raise ValueError()
    assert len(x) == len(v_smoothed)
    if trim_front > 0: v_smoothed[:trim_front] = np.nan
    return v_smoothed

def rem_nans(arr):
    """
    Removes nan elements from a numpy array or list. arr is modified in place.

    Parameters
    ----------
    arr : array-like or list of length n

    Returns
    -------
        None (arr is modified in place.)
    """
    if type(arr) is pd.core.frame.DataFrame: raise ValueError('Passed a pandas instead of numpy array')
    if type(arr).__module__ == np.__name__ : return arr[np.isfinite(arr)]
    elif type(arr) is list: return [xx for xx in arr if not np.isnan(xx)]
    else: raise TypeError(type(arr))


def sliding_window(seq, window_size, lmbda=lambda x: x):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

        print(list(range(1, 7))) ->                 [1, 2, 3, 4, 5, 6]
        print(list(sliding_window(range(1, 7), 3))) -> [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
        print(list(sliding_window(range(1, 7), 3, lmbda=np.nanmean))) -> [2.0, 3.0, 4.0, 5.0]
       """
    it = iter(seq)
    result = tuple(itertools.islice(it, window_size))
    if len(result) == window_size:
        yield lmbda(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield lmbda(result)


def dynamic_sliding_window_generator(seq, window_size, step_size, lmbda=None, trim_end=True):
    """ Same as sliding window above, except it can handle different step sizes.
        Each Window is guaranteed to be size window_size if trim_end is True. Else, we window can
        be smaller.
    """
    for i in range(0, len(seq), step_size):
        window = tuple(seq[i: i + window_size])
        if len(window) == window_size or not trim_end:
            if lmbda is None: yield window
            else: yield lmbda(window)

def dynamic_sliding_window(seq, window_size, step_size, from_back=False, lmbda=None, trim_end=True):
    """ Same as sliding window above, except it can handle different step sizes.
        Each Window is guaranteed to be size window_size if trim_end is True. Else, we window can
        be smaller.

        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window_size = 3
        from_back =
        trim_end = Each Window is guaranteed to be size window_size if trim_end is True. Else, we window can be smaller.
        windows = putil_array.dynamic_sliding_window(seq, window_size, step_size=window_size, from_back=from_back, lmbda=None, trim_end=trim_end)
        windows -> [ [2, 3, 4], [5, 6, 7 ], [7, 9, 10] ]

    """
    if from_back:
        seq = list(reversed(seq))
    res = list(dynamic_sliding_window_generator(seq, window_size, step_size, lmbda=lmbda, trim_end=trim_end))
    if from_back:
        res = [list(reversed(x)) for x in res]
        res = list(reversed(res))
    return res


def sliding_window_index_series(seq, window_size, session_pd=None, num_ts_per_trial=None, from_back=False, lmbda=None, trim_end=True, append_val=np.nan):
    """
    Takes a dynamic sliding window on the provided sequence
    :param seq: Takes a sequence
    :param window_size:
    :param session_pd:
    :param num_ts_per_trial: Can be passed directly or computed from session_pd
    :param from_back:
    :param lmbda:
    :param trim_end:
    :param append_val:
    :return:
    """
    if session_pd is not None:
        num_ts_per_trial = session_pd.groupby('trial_index').size().values
        num_ts_per_trial = num_ts_per_trial.astype(np.int)

        # num_ts_per_trial is a vector that contains the number of timesteps in each trial
        # ie:
        #   Trial 0 -> 10 timesteps
        #   Trial 1 -> 11 timesteps
        #  num_ts_per_trial = [10, 11]
    assert len(num_ts_per_trial) == len(seq)
    if num_ts_per_trial is None: raise ValueError(num_ts_per_trial)
    # For example: seq     -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    windows = dynamic_sliding_window(seq, window_size, step_size=window_size, from_back=from_back, lmbda=lmbda, trim_end=trim_end)
    # windows -> [[2, 3, 4], [5, 6, 7], [7, 9, 10]] (from_back = True)
    window_indexes = np.asarray([ii for ii in range(len(windows))])
    # window_indexes -> [0, 1, 2] (from_back = True)
    window_indexes_by_trial =    np.repeat(window_indexes, window_size, axis=0)
    # window_indexes_by_trial -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
    if len(window_indexes_by_trial) < len(num_ts_per_trial):
        if from_back: # append append_val (typically nan) to the front.
            # window_indexes_by_trial -> [nan, nan, 0, 0, 0, 1, 1, 1, 2, 2, 2]
            window_indexes_by_trial = np.append([append_val] * (len(num_ts_per_trial) - len(window_indexes_by_trial)), window_indexes_by_trial)
        else: # append append_val (typically nan) to the back.
            window_indexes_by_trial = np.append(window_indexes_by_trial, [append_val] * (len(num_ts_per_trial) - len(window_indexes_by_trial)))
    assert len(window_indexes_by_trial) == len(num_ts_per_trial), f'{len(window_indexes_by_trial)}   {len(num_ts_per_trial)}'
    assert len(num_ts_per_trial) == len(seq)
    window_indexes_by_timestep = np.repeat(window_indexes_by_trial, num_ts_per_trial, axis=0)
    # window_indexes_by_timestep is window_indexes_by_trial expanded out to have one entry per timestep in each trial.
    assert len(window_indexes_by_timestep) == np.sum(num_ts_per_trial), f'{len(window_indexes_by_timestep)}, {np.sum(num_ts_per_trial)}' # Assert have one entry per timestep
    assert window_indexes_by_timestep.ndim == 1
    if session_pd is not None:
        # another check to ensure that we have one entry per timestep.
        assert len(window_indexes_by_timestep) == len(session_pd), f'{len(window_indexes_by_timestep)}, {len(session_pd)}, {sum(num_ts_per_trial)}'
    return window_indexes_by_timestep
    
    
    
def get_list_partition_boundaries(lst, num_bins):
    """
    Takes a list and splits it up into num_bins compartments. This is done based on the maximum
    and minimum value in the list.

    get_intervals(list(range(11)), 3) -> min_intervals: [0, 3.3333333333333335, 6.666666666666667]
                                         max_intervals: [3.3333333333333335, 6.666666666666667, 10.0]
                                         tup_intervals: [(0, 3.3333333333333335), (3.3333333333333335, 6.666666666666667), (6.666666666666667, 10.0)]

    :param lst: list to get the intervals on
    :param num_bins: number of bins to split the list into
    :return:
        min_intervals: The left edges of the intervals
        max_intervals:
        tup_intervals:
    """
    minb, maxb = min(lst), max(lst)  # Min and max of our lists
    delta = maxb - minb  # Calculate the difference
    step_size = delta / num_bins  # Take the range of our list and divide it into little the number of bins
    min_intervals = list(np.arange(minb, maxb, step_size))
    max_intervals = []
    for min_interval in min_intervals:
        max_intervals.append(min_interval + step_size)

    # Lists should be equal in length and equal to the number of bins we are examining.
    # asserts.assert_equal(len(min_intervals), len(max_intervals))
    # asserts.assert_equal(len(min_intervals), num_bins)
    # Check to ensure our intervals properly capture all our data.
    assert min_intervals[0] <= lst[0]
    assert max_intervals[-1] >= lst[-1]
    tup_intervals = [(min_interval, max_interval) for min_interval, max_interval in zip(min_intervals, max_intervals)]
    return min_intervals, max_intervals, tup_intervals


def calc_most_extreme_window(x, window_size, compare, constraint_vector=None, constraint_val=None, x_index=None):
    """
    Takes a sliding window over the given list and returns the
    interval with the highest average value.

    lst = [4, 2, 3, 1, 0, 5]
    get_interval_with_highest_average(lst, 3) -> 0, 3
                            lst[0, 3] -> [4, 2, 3]

    lst = [3, 2, 1, 4, 0, 5]
    get_interval_with_highest_average(lst, 3) -> 3, 6
                            lst[0, 3] -> [4, 0, 5]

    :param x: List to find the interval over. lst[best_start_index:best_end_index] gives the window
    :param window_size: Size of the interval.
    :return: best_start_index: Inclusive start index.
             best_end_index: Exclusive End index
             best_window: lst[best_start_index: best_end_index]  (of length interval_size)
    """
    if x_index is None: x_index = [0]
    x = np.squeeze(np.asarray(x))
    assert x.ndim == 1
    if constraint_vector is None: constraint_vector = np.ones_like(x)
    if constraint_val is None: constraint_val = 0
    assert x.shape == constraint_vector.shape, f'{x.shape, constraint_vector, constraint_vector.shape}'
    # if type(fn) is str:
    assert compare in {'largest', 'smallest'}
    if compare == 'largest':
        compare = operator.ge # "gt(a, b) is equivalent to a > b"
        best_interval_avg = float('-inf')
    else:
        compare = operator.le
        best_interval_avg = float('inf')
    if len(x) < window_size: raise BrokenPipeError()

    best_start_index = np.nan
    for start_index in range(0, len(x) - window_size + 1):  # Need to be sure to include the last value
        window = x[start_index: start_index + window_size]
        constraint = constraint_vector[start_index: start_index + window_size]
        assert window_size == len(window)
        if np.any(np.isfinite(window)): interval_avg = np.nanmean(window)
        else: interval_avg = np.nan
        if np.isfinite(interval_avg) and compare(interval_avg, best_interval_avg):
            if np.sum(constraint) > constraint_val:
                best_start_index = start_index
                best_interval_avg = interval_avg
    if np.all(np.isnan(best_start_index)):
        return np.nan, np.nan, False
    else:
        best_end_index = best_start_index + window_size + x_index[0]
        best_start_index = best_start_index + x_index[0]
        best_window = x[best_start_index: best_end_index]
        return best_start_index, best_end_index, best_window
    
    
    
def safe_remove(lst, val_to_remove):
    """
    same as calling .remove, but doesn't throw an error if val to remove not in list
    """
    try: lst.remove(val_to_remove)
    except ValueError: pass
    return lst

def safe_remove_multiple(lst, vals_to_remove):
    """
    same as calling .remove, but doesn't throw an error if val to remove not in list
    """
    for l in vals_to_remove:
        safe_remove(lst, l)
    return lst

def remove_consecutive_items(lst):
    """ Works for any iterable including pandas Series. Does not modify lst
    Does not if there are nans in the list """
    return [x[0] for x in itertools.groupby(lst)]