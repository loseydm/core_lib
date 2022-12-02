import numpy as np
import scipy.stats

def univariate_ftest(x: np.ndarray, y: np.ndarray, p2: int=2):
    """
    Does a univariate_ftest on x and y. Tested to give the same results as Matlab and skikit-learn:
    
    Matlab ftest example:
        x = [3504;3693;3436;3433;3449;4341;4354;4312;4425;3850;3090;4142;4034;4166;3850;3563;3609;3353];
        y = [18;15;18;16;17;15;14;14;14;15;12;13;15;10;12;15;14;10];
        tbl = table(y,x);
        mdl = fitlm(tbl,'y ~ x')

    Scikit example:
        x = np.asarray([3504,3693,3436,3433,3449,4341,4354,4312,4425,3850,3090,4142,4034,4166,3850,3563,3609,3353])
        y = np.asarray([18,15,18,16,17,15,14,14,14,15,12,13,15,10,12,15,14,10])
        print(sklearn.feature_selection.f_regression(x.reshape(-1, 1), y))

    Parameters
    ----------
    x : array-like, shape (n,)
        input variable
    y : array-like, shape (n,)
        response variable
    p2 : number of parameters. Should always be 2 (y = mx + b), where m and b are the
         two parameters. Included for both clarity and in case I extend this to the multidimensional setting.

    Returns
    -------
    F : float
        F statistic
    pval : float 
        pvalue
    """    
    if len(x) != len(y) or np.any(~np.isfinite(x)) or not np.any(~np.isfinite(y)):
        raise ValueError('Invalid input')
    slope, intercept, r, p, se = scipy.stats.linregress(x, y)
    y2_pred = x * slope + intercept # unconstrained hypothesis
    y1_pred = np.mean(y) # constrained hypothesis
    n = len(y)
    rss2 = np.sum( (y - y2_pred) ** 2)
    rss1 = np.sum( (y - y1_pred) ** 2)
    p2 = 2 # number of parameters for unconstrained hypothesis
           # namely, y = mx + b, where m and b are the two parameters
    p1 = 1 # number of parameters for our constrained hypothesis (y = b)
    F = ((rss1 - rss2) / (p2 - p1)) / (rss2 / (n - p2 ))
    pval = scipy.stats.f.sf(F, dfn=1, dfd=n - p2)
    return F, pval



def bernoulli_post(dat, a=1, b=1, conf=0.95):
    """
    Given a vector of zeros and ones, computes the posterior mean and variance  of probability p.    
           p ~ Beta(a, b)
     x_i | p ~ Bernoulli(p)
    p | {x} ~ Beta(a + number of successes, b + number of failures)

    Default is uniform prior over [0, 1]
    Parameters
    ----------
    dat : data vector of zeros and ones

    Returns
    -------
        m          - posterior mean of p
        l          - lower limit of confidence interval, as specified by 'conf'
        u          - upper limit of confidence interval, as specified by 'conf'
        anew, bnew - parameters of posterior Beta distribution
    """
    anew = a + np.sum(dat)
    bnew = b + len(dat) - np.sum(dat)
    posterior_mean = anew / (anew + bnew)
    edg = (1-conf) / 2
    lower_bound = scipy.special.betaincinv(anew, bnew, edg)
    upper_bound = scipy.special.betaincinv(anew, bnew, 1 - edg)
    # posterior variance
    # v = anew * bnew / (anew + bnew) ** 2 / (anew + bnew + 1)
    return posterior_mean, lower_bound, upper_bound, anew, bnew