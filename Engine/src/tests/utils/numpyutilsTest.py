'''
Created on Mar 6, 2014

@author: xapharius
'''
import unittest
import utils.numpyutils as nputils
import numpy as np
import sys
from _ast import Assert

class numpyutilsTest(unittest.TestCase):

    
    def testaddOneToVec(self):
        '''
        Test valid execution of one-dim, row and column vector
        '''
        #vector array
        vec = np.array([1,2,3])
        vecOne = nputils.add_one_to_vec(vec)
        assert vecOne.shape[0] == 1 and vecOne.shape[1] == vec.shape[0]+1, "anddOneToVec fails for normal vec (x,)"
        
        #row vector array
        vec = np.array([[1,2,3]])
        vecOne = nputils.add_one_to_vec(vec)
        assert vecOne.shape[0] == 1 and vecOne.shape[1] == vec.shape[1]+1, "anddOneToVec fails for row vec (1,x)"
        
        #row vector array
        vec = np.array([[1],[2],[3]])
        vecOne = nputils.add_one_to_vec(vec)
        assert vecOne.shape[1] == 1 and vecOne.shape[0] == vec.shape[0]+1, "anddOneToVec fails for row vec (1,x)"
        
    def testaddOneToVec_neg(self):
        '''
        Test exception in case of matrix as parameter
        '''
        
        mat = np.array([[1,2,3],[4,5,6]])
        try:
            nputils.add_one_to_vec(mat)
            assert(False)
        except:
            expected_errmsg = "addOne works only for one dimensional vectors"
            errmsg = str(sys.exc_info()[1])
            assert(errmsg.startswith(expected_errmsg))
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()