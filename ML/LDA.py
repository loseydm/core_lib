import numpy as np

def calc_lda(X: np.ndarray, y: np.ndarray, k: int=None, equal_priors: bool=True) -> np.ndarray:
    """
    Applies Fisher linear discriminant analysis
    
    w = argmax w^T SB w / w^T SW w where
        S_w is the within class sum of squares.
        S_b is the between class sum of squares.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Input data, where n is the number of datapoints and d is number of features
    y : array-like, shape (n,)
        corresponding labels 
    k : int
        number of dimensions to reduce to.
        Defaults to c - 1, where c is the number of classes

    Returns
    -------
    W: a dxk numpy array
    """
    if k is None: k = len(set(y)) - 1
    S_w, S_b = _calc_LDA_Sw_Sb(X, y)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w) @ S_b)
    # sort the eigenvectors based on eigenvalue in ascending order
    w = eig_vecs[:, np.argsort(np.real(eig_vals))[::-1]]    
    return w[:, :k]


def _calc_LDA_Sw_Sb(X: np.ndarray, y: np.ndarray, equal_priors: bool=True) -> (np.ndarray, np.ndarray):
    """
    Calculates the within and between class sum of squares for use in LDA.

    Parameters
    ----------
    X : nxd. Number of datapoints by number of features
    y : n class labels

    Returns
    -------
    S_w : The within class sum of squares.
    S_b : The between class sum of squares.
    """
    n, d = X.shape    
    class_set = np.unique(y)
    c = len(class_set)
    
    #  [ [ mean vector of classs 1]
    #    [ mean vector of classs 2] ...]
    m = np.zeros(shape=(c, d) )    
    S = np.zeros(shape=(d, d, c))
    for class_idx, class_val in enumerate(class_set):
        v = X[y == class_val, :] # n x d
        nc = v.shape[0] # number of data points in class c
        if equal_priors: nc = n / c
        mean = np.mean(v, axis=0)
        m[class_idx, :] = mean
        S[:, :, class_idx] = nc * np.cov(  (v - mean).T, bias=True)
        np.testing.assert_almost_equal((v - mean).T @ (v - mean), nc * np.cov(  (v - mean).T, bias=True), decimal=5 )
        S[:, :, class_idx] = (v - mean).T @ (v - mean)
    S_w = np.sum(S, axis=-1) # dxd
    assert S_w.shape == (d, d)
    S_w = (S_w + S_w.T) / 2 # numerical stablitiy. S_w should be symmetric, but may not be due to numerical impercision.
    # m is class x features. mi is m[i, :].
    overall_mean = np.mean(X, axis=0).reshape((d, 1))
    assert overall_mean.shape == (d, 1)
    S_b = np.zeros((d,d))
    for class_idx, class_val in enumerate(class_set):
        v = X[y == class_val, :] # n x d
        nc = v.shape[0] # number of data points in class c
        if equal_priors: nc = n / c
        mean = m[class_idx, :].reshape((d, 1))
        S_b += nc * (mean - overall_mean) @ (mean - overall_mean).T    
    S_b = (S_b + S_b.T) / 2 # numerical stablitiy. S_b is already symmetric.
    assert S_b.shape == (d, d)
    return S_w, S_b # both are dxd matrices.

def _test_lda():
    """ A quick test of the calc_lda function """
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    feature_dict = {i:label for i,label in zip(
                    range(4),
                      ('sepal length in cm',
                      'sepal width in cm',
                      'petal length in cm',
                      'petal width in cm', ))}
    
    df = pd.io.parsers.read_csv(
        filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        sep=',',
        )
    df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
    df.dropna(how="all", inplace=True) # to drop the empty line at file-end
    X = df.iloc[:, [0,1,2,3]].values # x is nxd
    y = df['class label'].values
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1
    # W should be:
    W_target = np.asarray( [[-0.2049, -0.009 ],
                            [-0.3871, -0.589 ],
                            [ 0.5465,  0.2543],
                            [ 0.7138, -0.767 ]]
                          )
    W = calc_lda(X, y)
    np.testing.assert_allclose(W, W_target, atol=0.0001)

