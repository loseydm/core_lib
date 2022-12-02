import numpy as np
import scipy

def factor_analysis(X: np.ndarray, m: int, mode: str='FA', num_iterations: int=100):
    """
    Fits a factor analysis/probabilistic principal component analysis model.

    PPCA model:
        P(z) ~ N(z | 0, I)
        P(X|Z) ~ N(X | Wz + mu, sigma^2I)
        
    FA model:
        P(z) ~ N(z | 0, I)
        P(X|Z) ~ N(X | Wz + mu, psi)
    
    where:
        X is a nxd data matrix, where n is number of datapoints with and d is number of features.
        Z is a nxm factor matrix, where m is the number of latent factors 
        W is the dxm weight matrix
        sigma^2 is a scaler
        I is the dxd identity matrix
    
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix        
    m : int, number of latent factors
    mode : str, optional
        Either 'FA' for factor analysis or 'PPCA' for prob. pca. The default is FA.
    num_iterations : int, optional
        Number of EM steps to take. The default is 100.

    Returns
    -------
    W : np.ndarray, shape (n, d)
        Latent factors (note that they may not be orthonormalized)
    z : np.ndarray, shape (n, d)
        Latent factors (note that they may not be orthonormalized)
    LLs : list, len(num_iterations)
        List of log likelihoods recorded during training
    C : WW^T + psi, a 

    """
    mode = mode.upper()
    if mode not in {'FA', 'PPCA'}: raise ValueError('Invalid input {mode}, please select FA or PPCA')
    x_noncentered = X
    n, d = x_noncentered.shape
    X = (x_noncentered - np.mean(x_noncentered, axis=0)).T
    W = np.random.rand(d, m)
    psi = np.diag(np.diag(np.cov(X.T, rowvar=False))) # make off diagonal equal to 0
    sigma2 = 1.0
    LLs = []
    for i in range(num_iterations):
        ### e-step
        if mode == 'FA':     C = W @ W.T + psi
        elif mode == 'PPCA': C = W @ W.T + sigma2 * np.eye(d)
        E_zx = W.T @ np.linalg.inv(C) @ X # E[z|x]
        cov_zx = np.eye(m) - W.T @ np.linalg.inv(C) @ W
        ### m-step
        W = X @ E_zx.T @ np.linalg.inv(cov_zx * n + E_zx @ E_zx.T) 
        if mode == 'FA':      psi    = np.diag(np.diag(X @ X.T - W @ (E_zx @ X.T))) / n 
        elif mode == 'PPCA':  sigma2 = (1 / (n * d)) * np.trace(X @ X.T - W @ (E_zx @ X.T)) 
        ### Log - Likelihood
        if mode == 'FA':     sigma_x_given_z = W @ W.T + psi
        elif mode == 'PPCA': sigma_x_given_z = W @ W.T + sigma2 * np.eye(d) 
        sign, log_det = np.linalg.slogdet(sigma_x_given_z)
        log_det = log_det * sign
        coeff = -0.5 * (np.log(2 * np.pi) * d + log_det)
        LL = np.trace(coeff - 0.5 * (X.T @ np.linalg.inv(sigma_x_given_z)) @ X)
        LLs.append(LL)
    if mode == 'FA':     C = W @ W.T + psi
    elif mode == 'PPCA': C = W @ W.T + sigma2 * np.eye(d) 
    z = W.T @ np.linalg.inv(C) @ X
    return W, z, LLs, C



def pca(X, m):
    """
    Principal component analysis

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix        
    m : int, number of latent factors. If None, defaults to d

    Returns
    -------
    evects: np.ndarray, shape (m, d), the eigenvectors
    evals: np.ndarray, shape (m,), the eigenvalues
    latents: np.ndarray, shape (n, m), low d representation of data
    mu: np.ndarray, shape (d,), mean used for data centering
    """
    n, d = X.shape
    if m is None: m = d
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    assert X_centered.shape == (n, d)
    assert mu.shape == (d,)
    U, S, VT = scipy.linalg.svd(np.linalg.cov(X, bias=True))
    V = VT.T
    evals = np.sqrt((S ** 2) / n)
    evects = V[:, :m]
    latents = X_centered @ evects
    return evects, evals, latents, mu
