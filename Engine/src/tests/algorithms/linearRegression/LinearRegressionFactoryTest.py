'''
Created on Dec 14, 2013

@author: xapharius
'''
import unittest
import numpy as np
from algorithms.linearRegression import LinearRegressionFactory
from algorithms.linearRegression import LinearRegression

from numpy.testing.utils import assert_equal
from numpy.ma.testutils import assert_array_equal

class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        
    def test_constructor(self):
        paramSize = 2;
        linRegFactory = LinearRegressionFactory(paramSize)
        assert_equal(paramSize, linRegFactory.nrLRparams)
        assert_equal(linRegFactory.nrModels, 0)

    def test_get_instance(self):
        '''test if return type of get_instance is correct
        '''
        linRegFactory = LinearRegressionFactory(2)
        linReg1 = linRegFactory.get_instance()
        assert_equal(linRegFactory.nrModels, 1);
        assert_equal(type(linReg1), LinearRegression)
        
    def test_aggregate(self):
        ''' test if aggregation, by averaging, works for simple example
        '''
        linRegFactory = LinearRegressionFactory(3)
        linRegArr = np.array([])
        
        linReg1 = linRegFactory.get_instance()
        linReg1.set_params(np.array([1,1,1]))
        linRegArr = np.append(linRegArr, linReg1)
        
        linReg2 = linRegFactory.get_instance()
        linReg2.set_params(np.array([3,3,3]))
        linRegArr = np.append(linRegArr, linReg2)
        
        superReg = linRegFactory.aggregate(linRegArr)
        assert_array_equal(superReg.params, np.array([2,2,2]))
        assert_equal(linRegFactory.nrModels, 2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()