'''
Created on Dec 14, 2013

@author: xapharius
'''
import unittest
import numpy as np
from algorithms.linearRegression import LinearRegressionFactory
from algorithms.linearRegression import LinearRegression
from mrjob.protocol import JSONProtocol
from numpy.testing.utils import assert_equal
from numpy.ma.testutils import assert_array_equal

class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        
    def test_constructor(self):
        nrInputVars = 2;
        linRegFactory = LinearRegressionFactory(nrInputVars)
        assert_equal(nrInputVars+1, linRegFactory.nrLRparams)

    def test_get_instance(self):
        '''test if return type of get_instance is correct
        '''
        linRegFactory = LinearRegressionFactory(2)
        linReg1 = linRegFactory.get_instance()
        assert_equal(type(linReg1), LinearRegression)
        
    def test_aggregate(self):
        ''' test if aggregation, by averaging, works for simple example
        '''
        linRegFactory = LinearRegressionFactory(2)
        linRegArr = []
        
        linReg1 = linRegFactory.get_instance()
        linReg1.set_params(np.array([[1,1,1]]))
        linRegArr.append(linReg1)
        
        linReg2 = linRegFactory.get_instance()
        linReg2.set_params(np.array([[3,3,3]]))
        linRegArr.append(linReg2)
        
        superReg = linRegFactory.aggregate(linRegArr)
        assert_array_equal(superReg.params, np.array([[2,2,2]]))
        
    def test_encode(self):
        linRegFactory = LinearRegressionFactory(11)
        linReg = linRegFactory.get_instance()
        encoded = linRegFactory.encode(linReg)
    
        protocol = JSONProtocol()
        protocol.write(0, encoded)
        
    def test_decode(self):
        linRegFactory = LinearRegressionFactory(11)
        linReg = linRegFactory.get_instance()
        obj_encoded = linRegFactory.encode(linReg)
    
        protocol = JSONProtocol()
        json_encoded = protocol.write(0, obj_encoded)
        obj_encoded = protocol.read(json_encoded)
        
        linRegArr = linRegFactory.decode([obj_encoded])
        assert type(linRegArr) == list, "decoded not as a list"
        assert type(linRegArr[0]) == LinearRegression, "decoded not as LinearRegression"
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()