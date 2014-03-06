'''
Created on Mar 6, 2014

@author: xapharius
'''
import numpy as np

def addOneToVec(nparr):
    '''
    add a one to row or column vec(array)
    @param nparr: np.Array
    @rtype: np.Array 
    @raise exception: nparr hase not a vector shape
    '''
    
    # nparr is a (one dimensional) vector 
    if len(nparr.shape) == 1:
        t_nparr = np.append(nparr, 1)
    # row vector (array)
    elif nparr.shape[0] == 1:   
        t_nparr = np.append(nparr, [[1]], 1)
    # column vector (array)
    elif nparr.shape[1] == 1:
        t_nparr = np.append(nparr, [[1]], 0)
    else:
        raise Exception("addOne works only for one dimensional vectors")
        
    return t_nparr
    
