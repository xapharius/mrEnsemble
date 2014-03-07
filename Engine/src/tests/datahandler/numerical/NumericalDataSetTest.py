'''
Created on Mar 6, 2014

@author: xapharius
'''
import unittest
import numpy as np
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from numpy.ma.testutils import assert_equal

class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        inputs = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        labels = np.array([[0],[1],[2],[3],[4]])
        self.dataSet = NumericalDataSet(inputs, labels)
        self.nrObs = 5

    def test_get_observation(self):
        '''
        Test getting a single observation as (1,x) numpy array from input dataset
        '''
        for i in range(self.nrObs):
            inputs, labels = self.dataSet.get_observation(i)
            assert_equal(inputs, i*np.ones((1,3)), 'wrong input at observation %d'%i)
            assert_equal(labels, i*np.ones((1,1)), 'wrong label at observation %d'%i)
            
    def test_gen_observations(self):
        '''
        Test generator getting all observations as (1,x) numpy array from input dataset
        '''
        i = 0
        for inputs, labels in self.dataSet.gen_observations():
            assert_equal(inputs, i*np.ones((1,3)), 'wrong input at observation %d'%i)
            assert_equal(labels, i*np.ones((1,1)), 'wrong label at observation %d'%i)
            i = i + 1

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()