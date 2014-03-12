'''
Created on Mar 6, 2014

@author: xapharius
'''
import numpy as np
import math

def addOneToVec(npArr):
    '''
    add a one to row or column vec(array)
    @param nparr: np.Array
    @rtype: 2-dim np.Array 
    @raise exception: nparr hase not a vector shape
    '''
    
    # nparr is a (one dimensional) vector 
    if len(npArr.shape) == 1:
        t_npArr = np.append(npArr, 1)
        t_npArr = np.reshape(t_npArr, (1,len(t_npArr)))
    # row vector (array)
    elif npArr.shape[0] == 1:   
        t_npArr = np.append(npArr, [[1]], 1)
    # column vector (array)
    elif npArr.shape[1] == 1:
        t_npArr = np.append(npArr, [[1]], 0)
    else:
        raise Exception("addOne works only for one dimensional vectors")
        
    return t_npArr
    
    
def sigmoidScalar(x):
    '''
    @param x: scalar
    @return: sigmoid of x
    @rtype: scalar
    '''
    return 1 / (1 + math.exp(-x))

def sigmoidNPArray(npArr):
    '''
    @param npArr: np.ndarray
    @return: sigmoid of x
    @rtype: np.ndarray
    '''
    func = np.vectorize(sigmoidScalar)
    return func(npArr)

def sigmoidDerivScalar(x):
    '''
    Given a scalar (e.g the weighted sum) compute the derivative of its sigmoid
    @param x: scalar
    @return: derivative of sigmoid of x
    @rtype: scalar
    '''
    sigX = sigmoidScalar(x)
    return sigX * (1- sigX)

def sigmoidDerivNPArray(npArr):
    '''
    Given a np.array (e.g the weighted sums) compute the derivative of its sigmoids
    @param npArr: np.ndarray
    @return: derivative of sigmoid of x
    @rtype: np.ndarray
    '''
    func = np.vectorize(sigmoidDerivScalar)
    return func(npArr)

def expNPArrayList(lst, power):
    '''
    Raise elements of a list (of np.arrays - elementwise) to power
    @param lst: list of np.arrays
    @param power: scalar, to which power the elements should be raised
    @rtype list of np.arrays
    '''
    retList = []
    for nparr in lst:
        retList.append(nparr ** power)
     
    return retList