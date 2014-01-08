'''
Created on Dec 14, 2013

@author: xapharius
'''
import unittest
import numpy as np
from algorithms.linearRegression import *

from numpy.testing.utils import assert_equal
from numpy.ma.testutils import assert_array_equal

class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        
    def test_constructor(self):
        paramSize = 2;
        linRegFactory = LinearRegressionFactory(paramSize)
        assert_equal(paramSize, linRegFactory.nrLRparams)
        assert_equal(0, len(linRegFactory.linRegArr))

    def test_get_instance_correctReturn(self):
        '''test if return type of get_instance is correct
        '''
        linRegFactory = LinearRegressionFactory(2)
        linReg1 = linRegFactory.get_instance()
        assert_equal(linReg1, linRegFactory.linRegArr[0]);
        assert_equal(type(linReg1), LinearRegression)
        
    def test_get_instance_linRegArr(self):
        '''test if factory model array contains all created instances
        '''
        nrInstances = 10
        linRegArr = np.array([])
        linRegFactory = LinearRegressionFactory(2)
        for _ in range(nrInstances):
            linRegArr = np.append(linRegArr, linRegFactory.get_instance())
        assert_array_equal(linRegArr, linRegFactory.linRegArr)
        
    def test_aggregate(self):
        ''' test if aggregation, by averaging, works for simple example
        '''
        linRegFactory = LinearRegressionFactory(3)
        linReg1 = linRegFactory.get_instance()
        print linReg1.params
        linReg1.set_params(np.array([1,1,1]))
        linReg2 = linRegFactory.get_instance()
        linReg2.set_params(np.array([3,3,3]))
        superReg = linRegFactory.aggregate()
        assert_array_equal(superReg.params, np.array([2,2,2]))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()