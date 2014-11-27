"""
Created on Mar 6, 2014

@author: xapharius
"""
import numpy as np
import math
import sys


def add_one_to_vec(np_arr):
    """
    add a one to row or column vec(array)
    @param np_arr: np.Array
    @rtype: 2-dim np.Array 
    @raise exception: np_arr has not a vector shape
    """
    
    # nparr is a (one dimensional) vector 
    if len(np_arr.shape) == 1:
        t_np_arr = np.append(np_arr, 1)
        t_np_arr = np.reshape(t_np_arr, (1, len(t_np_arr)))
    # row vector (array)
    elif np_arr.shape[0] == 1:
        t_np_arr = np.append(np_arr, [[1]], 1)
    # column vector (array)
    elif np_arr.shape[1] == 1:
        t_np_arr = np.append(np_arr, [[1]], 0)
    else:
        raise Exception("addOne works only for one dimensional vectors")
        
    return t_np_arr


def sigmoid_scalar(x):
    """
    @param x: scalar
    @return: sigmoid of x
    @rtype: scalar
    """
    if x > 100:
        # sys.stderr.write("sigmoid received value greater 100: " + str(x) + "\n")
        return 1.
    elif x < -100:
        # sys.stderr.write("sigmoid received value smaller -100: " + str(x) + "\n")
        return 0.
    return 1. / (1. + math.exp(-x))


def sigmoid_np_arr(np_arr):
    """
    @param np_arr: np.ndarray
    @return: sigmoid of x
    @rtype: np.ndarray
    """
    func = np.vectorize(sigmoid_scalar)
    return func(np_arr)


def sigmoid_deriv(x):
    """
    Calculates the sigmoid derivative for the given value.
    :param x: Values whose derivatives should be calculated
    :return: Derivatives for given values
    """
    return x * (1. - x)


def sigmoid_deriv_scalar(x):
    """
    Given a scalar (e.g the weighted sum) compute the derivative of its sigmoid
    @param x: scalar
    @return: derivative of sigmoid of x
    @rtype: scalar
    """
    sigx = sigmoid_scalar(x)
    return sigx * (1 - sigx)


def sigmoid_deriv_np_arr(np_arr):
    """
    Given a np.array (e.g the weighted sums) compute the derivative of its sigmoids
    @param np_arr: np.ndarray
    @return: derivative of sigmoid of x
    @rtype: np.ndarray
    """
    func = np.vectorize(sigmoid_deriv_scalar)
    return func(np_arr)


def exp_np_array_list(lst, power):
    """
    Raise elements of a list (of np.arrays - elementwise) to power
    @param lst: list of np.arrays
    @param power: scalar, to which power the elements should be raised
    @rtype list of np.arrays
    """
    result = []
    for nparr in lst:
        result.append(nparr ** power)
     
    return result


def to_list(arr):
    """
    Creates a list representation of the given np.array or list of np.arrays.
    @param arr: np.array or list of np.arrays
    @return: list representation of the given input
    @rtype: list 
    """
    try:
        return arr.tolist()
    except AttributeError:
        result = []
        for row in arr:
            result.append(row.tolist())
        return result


def tanh_deriv(y):
    """
    Calculate the derivative for the given tanh value (y = tanh(x))
    :param y: tanh function value
    :return: Derivative for given value
    """
    return 1 - np.power(y, 2)


def rot180(arr):
    """
    Rotates the given numpy array by 180 degree counter-clock-wise.
    :param arr: Numpy array to be rotated
    :return: 180 degree counter-clock-wise rotated numpy array
    """
    return np.rot90(np.rot90(arr))


def calc_squared_error(expected, actual):
    """
    Calculates the mean squared error for the given values.
    :param expected: Numpy array representing the target.
    :param actual: Numpy array of actual values.
    :return: Mean squared error
    """
    return np.sum(0.5 * np.power(expected - actual, 2))


def normalize_arr(arr, new_min, new_max):
    """
    Normalizes the given array.
    :param arr: Numpy array to be scaled.
    :param new_min: Minimum value for the resulting array.
    :param new_max: Maximum value for the resulting array.
    :return: Normalized numpy array a where a.min() == new_min and a.max() ==
    new_max
    """
    img_min = arr.min()
    img_max = arr.max()
    if img_min == img_max:
        return np.zeros(arr.shape)
    return (arr - img_min) * ((new_max - new_min) / (img_max - img_min)) + new_min


def vec_with_one(length, idx):
    """
    Creates a numpy vector of zeros except of the given idx that is set to 1.
    :param length: Length of created vector
    :param idx: Index that should be set to 1
    :return: Vector v of given length where v[idx]==1
    """
    vec = np.zeros(length)
    vec[idx] = 1
    return vec


def softmax(arr, t=1.0):
    """
    Softmax function that squashes the given vector's values so that all values
    are in the range 0 to 1 and sum to 1.
    :param arr: Numpy vector to be squashed
    :param t: Temperature
    :return: Numpy vector with values between 0 and 1, summing to 1.
    """
    e = np.exp(arr / t)
    return e / np.sum(e)


def convert_targets(targets, num_classes):
    """
    Converts classes represented by a single number to vectors of length
    num_class where the appropriate index is set to one, all others are set to
    zero.
    :param targets: List of targets
    :param num_classes: Overall number of classes
    :return: 2D numpy array where each row represents a target vector.
    """
    result = np.zeros((len(targets), num_classes))
    for i in range(len(targets)):
        result[i, targets[i]] = 1
    return np.array(result)