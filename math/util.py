import numpy as np

def histogram_match_vectors(x_matchby, y_matchby, x_eval, y_eval, num_bins):
    """
    Takes two vectors x_matchby and y_matchby and performs histogram matching 
    with the given number of bins.
    (i.e. x_matchby, y_matchby and distribution matched - they will have the same histogram
     shape when the returned bins are used.)
                                                                               )
    It finds what indexes to keep, then returns x_eval and y_eval with only those indexes.
    :param x_matchby: matchby vector (same length as eval vector)
    :param y_matchby: matchby vector (same length as eval vector)
    :param x_eval: evaluation vector (uses only indexes that are matched betweent he matchby vects).
    :param y_eval: evaluation vector (uses only indexes that are matched betweent he matchby vects).
    :param num_bins: Number of bins to use in the histogram matching.
    """
    if len(x_matchby) != len(x_eval): raise ValueError()
    if len(y_matchby) != len(y_eval): raise ValueError()
    pd1_SortBy_Progress = x_matchby  # What progress we want to filter by. Usually IntuProgress
    pd2_SortBy_Progress = y_matchby
    assert len(pd1_SortBy_Progress) > 0 and len(pd2_SortBy_Progress) > 0
    hist, bins = np.histogram(list(pd1_SortBy_Progress) + list(pd2_SortBy_Progress), bins=num_bins, density=False)  # Gives us a list of bin edges [left edge, left edge,... right edge].
    assert len(bins) == num_bins + 1  # Fence post problem. Bins mark our posts. num_bins mark our fence.
    pd1_inds = np.digitize(pd1_SortBy_Progress, bins)  # Get which bin we are in
    pd2_inds = np.digitize(pd2_SortBy_Progress, bins)
    assert len(pd1_SortBy_Progress) == len(pd1_inds)
    assert len(pd2_SortBy_Progress) == len(pd2_inds)
    bin_dict = dict() # Initialize our bin dictionary.
    for bin in bins: bin_dict[bin] = {'pd1': [], 'pd2': []}
    for jj, bin in enumerate(bins):
        bin_dict[bin]['pd1'] = np.where(pd1_inds == jj)[0]  # gives the index data points that fall into this bin
        bin_dict[bin]['pd2'] = np.where(pd2_inds == jj)[0]
        num_in_pd1 = len(bin_dict[bin]['pd1'])
        num_in_pd2 = len(bin_dict[bin]['pd2'])
        m = min(num_in_pd1, num_in_pd2)
        np.random.shuffle(bin_dict[bin]['pd1'])
        np.random.shuffle(bin_dict[bin]['pd2'])
        bin_dict[bin]['pd1'] = bin_dict[bin]['pd1'][:m]
        bin_dict[bin]['pd2'] = bin_dict[bin]['pd2'][:m]
    x_indexes_to_keep = np.asarray([])
    y_indexes_to_keep = np.asarray([])
    for jj, bin in enumerate(bins):
        x_indexes_to_keep = np.concatenate((x_indexes_to_keep, bin_dict[bin]['pd1']), axis=0)
        y_indexes_to_keep = np.concatenate((y_indexes_to_keep, bin_dict[bin]['pd2']), axis=0)
    y_eval = y_eval[sorted(list(y_indexes_to_keep.flatten().astype(np.int)))]
    x_eval = x_eval[sorted(list(x_indexes_to_keep.flatten().astype(np.int)))]
    assert len(x_eval) == len(y_eval)
    return x_eval, y_eval, bins, x_indexes_to_keep, y_indexes_to_keep