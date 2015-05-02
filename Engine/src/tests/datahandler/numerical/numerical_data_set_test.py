'''
Created on Mar 6, 2014

@author: xapharius
'''
import unittest
import numpy as np
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from numpy.ma.testutils import assert_equal

class NumericalDatasetTest(unittest.TestCase):
    
    def test_constructor_all_params(self):
        '''
        Test constructor for supplied inputs and labels with same nr observations
        '''
        inputs = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        labels = np.array([[0],[1],[2],[3],[4]])
        dataSet = NumericalDataSet(inputs, labels)
        assert dataSet.nrInputVars == 3
        assert dataSet.nrTargetVars == 1
        assert dataSet.nrObservations == 5
        
    def test_constructor_nr_observations_mismatch(self):
        '''
        Test constructor for supplied inputs and labels with different nr observations
        '''
        inputs = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        labels = np.array([[0],[1],[2],[3]])
        try:
            NumericalDataSet(inputs, labels)
        except Exception, errmsg:
            expected_errmsg = "number of inputs and targets observations mismatch"
            self.assertTrue(errmsg.message.startswith(expected_errmsg))
        else:
            self.fail("no Exception thrown")
            
            
    def test_get_observation(self):
        '''
        Test getting a single observation as (1,x) numpy array from input dataset
        '''
        inputs = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        labels = np.array([[0],[1],[2],[3],[4]])
        dataSet = NumericalDataSet(inputs, labels)
        nrObs = 5
        
        for i in range(nrObs):
            inputs, labels = dataSet.get_observation(i)
            assert_equal(inputs, i*np.ones((1,3)), 'wrong input at observation %d'%i)
            assert_equal(labels, i*np.ones((1,1)), 'wrong label at observation %d'%i)
            
    def test_get_observation_no_labels(self):
        '''
        get observations from a dataset without labels
        '''
        inputs = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        dataSet = NumericalDataSet(inputs)
        nrObs = 5
        for i in range(nrObs):
            inputs, target = dataSet.get_observation(i)
            assert target == None
            assert_equal(inputs, i*np.ones((1,3)), 'wrong input at observation %d'%i)    
            
    def test_gen_observations(self):
        '''
        Test generator getting all observations as (1,x) numpy array from input dataset
        '''
        inputs = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        labels = np.array([[0],[1],[2],[3],[4]])
        dataSet = NumericalDataSet(inputs, labels)
        
        i = 0
        for inputs, labels in dataSet.gen_observations():
            assert_equal(inputs, i*np.ones((1,3)), 'wrong input at observation %d'%i)
            assert_equal(labels, i*np.ones((1,1)), 'wrong label at observation %d'%i)
            i = i + 1
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()