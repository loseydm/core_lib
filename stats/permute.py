"""
Contains tools for performing permutation tests.
"""

import numpy as np

def permute_subset(x: np.ndarrray, to_permute_subset: np.ndarray, inplace: bool=False):
    """ 
    Takes array x and a list of elements to permute in that array. 
    Permutes only those elements, leaving the rest alone.
    
    Parameters
    ----------
    x : array-like, shape (n,)
        array to be permuted
    to_permute_subset : array-like, shape (c,) where c is number of elements to permute
        elements within x to permute, leaving other elements alone
    Example:
        x = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        to_permute_subset = np.asarray([1, 3])
        permute_subset(x, to_permute_subset, inplace=False) # -> array([0, 1, 2, 1, 4, 0, 3, 2, 3, 4])
    
    Returns
    -------
    x : array-like, shape (n,)
        If inplace, modifies x inplace.        
    """  
    x = np.squeeze(np.asarray(x))
    assert x.ndim == 1
    if not inplace: x = np.copy(x)
    permute_flag = np.isin(x, to_permute_subset)
    per_array = x[permute_flag]
    x[permute_flag] = np.random.permutation(per_array)
    return x