""" Contains standard neuroscience tools """

import numpy as np
import twod_geo
import pandas as pd
import warnings
import collections

def fit_cosine_tuning_curve(spike_counts, rad_angles):
    """ 
        Fits cosine tuning curves to a spike count matrix. 
        Tuning is of the form:
        
            λ(s) = $$r_0 + (r max − r_0 ) cos(s − s_{max} )$$
            r is the spiking rates
            s = theta in my code below
            λ is the spike counts.

        spike counts as a function of angle s = baseline_firing rate + amplitude cosine(s - smax)

    Terms:
        max_fr = rmax
        baseline_fr = r0
        predicted_spike_counts = lmbda
    """
    # ## Spike Counts = Angles x Spikes
    twod_geo.assert_in_radian_range(rad_angles)
    num_conditions = spike_counts.shape[0]
    if np.any(rad_angles > 2 * np.pi): raise ValueError('Angles May be in Degrees')
    if np.any(rad_angles < 0): warnings.warn('Negative Angles')
    lmbda = spike_counts
    assert lmbda.shape == (num_conditions,)

    """ A = [ones(num_cons,1), cos(s), sin(s)];
        x = A\lambda; % closed form solution that minimizes norm(A*x-lambda,2)
    """
    A = np.vstack([np.ones(num_conditions),
                   np.cos(rad_angles),
                   np.sin(rad_angles)]).T
    assert np.all(A[:, 0] == np.ones(num_conditions)) and (np.all(A[:, 1] == np.cos(rad_angles)))
    x, _, _, _ = np.linalg.lstsq(A, lmbda, rcond=None)
    a = x[1]
    b = x[2]
    # warnings.filterwarnings('error')
    try:    theta_max_rad = np.arctan(b / a)
    except:
        theta_max_rad = np.nan; print(f'{x[2], x[1]}')
        raise

    # if theta_max_rad < 0: theta_max_rad = - theta_max_rad


    a_coeff = a / np.cos(theta_max_rad)
    b_coeff = b / np.sin(theta_max_rad)
    try:
        np.testing.assert_allclose(a_coeff, b_coeff, atol=0.000000001)
    except:
        print('Assertion error', a_coeff, b_coeff)
    
    """ r_0 = x(1); % the offset on both sides of the equation must be same
        r_max = r_0+x(2)/cos(s_max);
    """
    r0 = x[0]
    rmax = r0 + x[1] / np.cos(theta_max_rad)
    if r0 > rmax:
        rmin = rmax
        rmax = 2 * r0 - rmin
        theta_max_rad = theta_max_rad + np.pi
    theta_max_rad = theta_max_rad % (2 * np.pi)

    """ s_max = atan(x(3)/x(2)) """
    assert 0 <= theta_max_rad <= 2 * np.pi, theta_max_rad
    assert rmax > 0
    pref_dir = theta_max_rad
    max_fr = rmax
    baseline_fr = r0
    predicted_spike_counts = lmbda
    return baseline_fr, max_fr, pref_dir, predicted_spike_counts


def fit_cosine(hist, spike_names):
    d = collections.defaultdict(list)
    for name in spike_names:
        s = hist.index.values
        xdata = hist[name].values
        r0, rmax, smax, lmbda = fit_cosine_tuning_curve(xdata, s)
        d['s_deg'].append(s)
        d['rmax'].append(rmax)
        d['smax'].append(smax)  # pref dir.
        d['r0'].append(r0)
        d['lmbda'].append(lmbda)
    df = pd.DataFrame(dict(d))
    df.index = spike_names
    return df

def cosine_tune_fn(angles, baseline_fr, max_fr, pref_dir):
    """ λ(s) = r 0 + (r max − r 0 ) cos(s − s max)
    s is the reaching angle of the arm (in degrees),
    smax (smax in radians) is the reaching angle associated with the maximum response rmax
    r0 is an offset that shifts the tuning curve up from the zero axis
        returns  lambda =     predicted_spike_counts


    Angles must be in radians.
    """
    s = angles
    smax = pref_dir
    rmax = max_fr
    r0 = baseline_fr
    twod_geo.assert_in_radian_range(s)
    predicted_spike_counts = lmbda = r0 + (rmax - r0) * np.cos(s - smax)
    return predicted_spike_counts

