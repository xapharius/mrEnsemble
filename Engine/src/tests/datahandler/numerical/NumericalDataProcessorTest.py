'''
Created on Jan 11, 2014

@author: xapharius
'''
import unittest
from datahandler.numerical import NumericalDataProcessor 
from numpy.ma.testutils import assert_equal
import numpy as np


class NumericalDataProcessorTest(unittest.TestCase):

    def setUp(self):
        self.rawData = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6]])


    def test_constructor(self):
        dataProc = NumericalDataProcessor(self.rawData, 3, 1)
        dataSet = dataProc.getData()
        assert_equal(dataSet.inputs.shape, (3,3))
        assert_equal(dataSet.labels.shape, (3,1))
    
    #TODO: implement test
    def test_normalize_data(self):
        assert(False);    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()