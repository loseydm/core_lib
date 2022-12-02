"""
Contains tools for performing operations on 2D vectors (e.g. rotation)

This file uses the convenction:
 theta_any: angle in degrees that can be form theta180, theta360
 theta180: -180 < theta <= 180
 theta360:  0 <= theta < 360
 
 thetas_rad: theta in radians

"""

import numpy as np
import warnings


def calc_rotation_matrix(theta, in_deg=True):
    """
    Calculates a rotatio matrix for the specified theta

    Parameters
    ----------
        theta : float
            angle for rotation matrix in degrees. Can be negative or bigger than 360 deg.
        in_deg : bool
            theta in degrees if true. else theta is in radians
    
    Returns
    -------
        R : array-like, shape (2, 2)
            Should be left multiplied.
            v' = Rv,
        Where v is a column vector.
    """
    if in_deg: theta = np.radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array(((c, -s),
                  (s, c)))
    return R

def calc_angle_between180(vects1, vects2, warn_on_zero_vect=True):
    """
    Calculates angles between vects1 and vects2 rowwise

    if vect2 is clockwise of vect1, then the result is positive
    if vect2 is counterclockwise of vect1, then the result is negative

    Parameters
    ----------
      vects1: array-like, shape (n, 1) or (n,)
          First set of vectors
      vects2: array-like, shape (n, 1) or (n,)
          second set of vectors
    """
    vects1 = convert_to_2Dvector(vects1)
    vects2 = convert_to_2Dvector(vects2)
    if vects1.shape != vects2.shape: raise ValueError()
    angles1 = vects_to_angles360(vects1, warn_on_zero_vect=warn_on_zero_vect)
    angles2 = vects_to_angles360(vects2, warn_on_zero_vect=warn_on_zero_vect)
    angular_diff180 = convert_arbitrary_to_theta180(angles2 - angles1)
    angular_diff180 = round_theta180(angular_diff180)
    return convert_singleton_to_float(angular_diff180)

def vects_to_angles180(vects, warn_on_zero_vect=True):
    """
        vects is shape Nx2
    """
    vects = convert_to_2Dvector(vects)
    xcoords = vects[:, 0]
    ycoords = vects[:, 1]
    is_zero_vect = np.logical_and(xcoords == 0, ycoords == 0)
    if warn_on_zero_vect and np.any(is_zero_vect): warnings.warn('Zero Vector Encountered')
    xcoords = np.where(is_zero_vect, np.nan,    xcoords)
    ycoords = np.where(is_zero_vect, np.nan, ycoords)
    thetas180 = np.rad2deg(np.arctan2(ycoords, xcoords))  # y is supposed to be the first argument.
    # numpy where format: numpy.where(condition, be this, else this).
    thetas180 = round_theta180(thetas180)
    assert_range_theta180(thetas180)
    return convert_singleton_to_float(thetas180)

def convert_arbitrary_to_theta180(arb_theta):
    """ converts """
    return convert_theta360_to_theta180(convert_arbitrary_to_theta360(arb_theta))

def assert_in_radian_range(thetas_rad):
    """ Asserts all angles are in [0, 2 * pi) """
    assert np.all((0 <= thetas_rad) & (thetas_rad < 2 * np.pi)), thetas_rad

def warn_if_in_radian_range(thetas_deg):
    """ warn if all angles in [-pi,    2pi)"""
    return 'Not doing this.'
    thetas_deg = convert_to_1Dvector(thetas_deg)
    if np.all((-np.pi <= thetas_deg) & (thetas_deg <= 2 * np.pi)):
        warnings.warn( 'Possibly Passed Radians Instead of Degrees')

def assert_in_degree_range(thetas_deg):
    """ Asserts all vectors past angle vectors is in [0, 360)
        Raises a warning if all angles are in [0, 2 * pi]
        :param thetas_deg: vector of angles  """
    thetas_deg = convert_to_1Dvector(thetas_deg)
    warn_if_in_radian_range(thetas_deg)
    assert np.all((0 <= thetas_deg) & (thetas_deg <= 360))

def convert_arbitrary_to_theta360(arb_theta):
    """ converts """
    arb_theta = convert_to_1Dvector(arb_theta)
    theta360 = arb_theta % 360  # Can handle angles in (-inf to inf)
    assert_range_theta360(theta360)
    return convert_singleton_to_float(theta360)

def round_theta180(theta180):
    """ Takes all values within rounding distance of -180 and changes it to be +180"""
    with warnings.catch_warnings(record=True) as w: # catch warnings for nan
        bools = (theta180 < -179.999999)
        if len(w) == 1: assert not np.all(np.isfinite(theta180))
    if np.any(bools):
        theta180 = np.where(bools, 180, theta180)  # numpy.where(condition, be this, else this).
    return theta180

def convert_to_1Dvector(data):
    """ Takes data in the form of a list, range, np array or number
        and returns data as a numpy array of shape (n,)
    """
    if type(data) is np.ndarray: pass
    elif type(data) is list: data = np.asarray(data)
    elif type(data) is range: data = np.asarray(data)
    elif isinstance(data, (int, float)): data = np.asarray([data])
    else: raise TypeError(f'Invalid Type: {type(data)}')
    assert data.ndim == 1, data.ndim
    return data


def _remove_nan(thetas): return thetas[np.isfinite(thetas)]

def assert_range_theta360(theta360, ignore_nan=True):
    """ theta360 is a vector of degrees"""
    theta360 = convert_to_1Dvector(theta360)
    warn_if_in_radian_range(theta360)
    if ignore_nan: theta360 = _remove_nan(theta360)
    assert np.all(theta360 != 360), np.nanmax(theta360)  # We don't include 360. This should be zero
    assert np.all((0 <= theta360) & (theta360 < 360))

def convert_theta360_to_theta180(theta360):
    """ takes theta360 and converts it to theta180 """
    assert_range_theta360(theta360)
    theta360 = convert_to_1Dvector(theta360)
    with warnings.catch_warnings(record=True) as w:
        theta180 = np.where(theta360 > 180, theta360 % -180, theta360)
        if len(w) == 1: assert not np.all(np.isfinite(theta360))
    theta180 = round_theta180(theta180)
    theta180 = convert_singleton_to_float(theta180)
    assert_range_theta180(theta180)
    return theta180

def assert_range_theta180(theta180, ignore_nan=True):
    """ theta180 is a vector of degrees"""
    theta180 = convert_to_1Dvector(theta180)
    warn_if_in_radian_range(theta180)
    if ignore_nan: theta180 = _remove_nan(theta180)
    # If we are in rounding distance to -180, we convert this to +180
    assert np.all((-180 < theta180) & (theta180 <= 180)), f'Ignore Nan?: {ignore_nan}, Min={np.min(theta180)}, Min={np.max(theta180)}'

def convert_theta180_to_theta360(theta180):
    """ Converts unsigned360 angles to a signed180 angle. Nans ignored.
        Can handle angles in (-inf to inf) (ie not limited to range of theta180, but I specifically test for
        the proper theta180 range.
    """
    theta180 = convert_to_1Dvector(theta180)
    assert_range_theta180(theta180)
    theta360 = theta180 % 360  # Can handle angles in (-inf to inf)
    theta360 = round_theta360(theta360)
    assert_range_theta360(theta360)
    return convert_singleton_to_float(theta360)

def round_theta360(theta360):
    """ Takes all values within rounding distance of 360 and changes it to be 0"""
    with warnings.catch_warnings(record=True) as w: # catch warnings for nan
        bools = (theta360 > 359.9999)
        if len(w) == 1: assert not np.all(np.isfinite(theta360))
    if np.any(bools):
        theta360 = np.where(bools, 0, theta360) # numpy.where(condition, be this, else this).
    return theta360

def vects_to_angles360(vects, warn_on_zero_vect=True):
    return convert_theta180_to_theta360(vects_to_angles180(vects, warn_on_zero_vect))

def angles_to_unit_vectors(thetaAny, round_small_epsilon=False):
    """ Works with theta360, theta180 and thetaAbs.
        Return shape N x 2 if thetaAny has more than one element or (2,) if thetaAny is a number
    """
    thetaAny = convert_to_1Dvector(thetaAny)
    x = np.cos(np.deg2rad(thetaAny))
    y = np.sin(np.deg2rad(thetaAny))
    unit_vect = np.asarray([x, y]).T
    assert unit_vect.ndim == 2
    assert unit_vect.shape[1] == 2
    if round_small_epsilon: unit_vect = _round_small_to_zero(unit_vect, epsilon=round_small_epsilon)
    return unit_vect.squeeze()

def _round_small_to_zero(a, epsilon=10**-12):
    """ Takes a matrix a (any shape).  All elements less than epsilon are set to zero. Modifies in place. """
    if type(epsilon) is bool: epsilon=10**-12
    a[np.abs(a) < epsilon] = 0
    return a

def convert_singleton_to_float(data):
    """ numpy array (any shape) -> numpy array unmodified
        numpy array of shape (1,) -> float
        int -> float
        float -> float
    """
    if type(data) is np.ndarray:
        if data.shape == (1,): return float(data)
        else: return data
    elif isinstance(data, (int, float)): return float(data)
    else: raise ValueError(data)

def convert_to_2Dvector(data):
    """ data: An n x 2 array or an array of shape (2,) [that is expanded to be (1, 2)]

        Takes data of shape (p,) or (n, p)
        if (n, p), returns data unmodified
        else returns data with shape (1, p)
        p must be equal to 2.
    """
    assert type(data) is np.ndarray, print(type(data))
    if ((data.ndim == 2) and (data.shape[1] != 2)) :
        raise AssertionError(f'Passed data should have been shape nx2. Data was {data.shape}')

    if data.ndim == 0: raise NotImplementedError()
    if data.ndim == 1: data = np.expand_dims(data, axis=0)
    if (data.shape[1] != 2 or data.ndim != 2):
        raise AssertionError(f'Data should have been shape nx2. Data was {data.shape}')
    return data
