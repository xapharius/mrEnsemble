'''
Created on Dec 4, 2013

@author: xapharius
'''
import unittest
import numpy as np
from algorithms.linearRegression.LinearRegression import LinearRegression


class LinearRegressionTest(unittest.TestCase):


    def setUp(self):
        class LocalDataSet(object):
            pass  
        dataSet = LocalDataSet()
        
        inputs = []
        targets = []
        for i in range(100):
            inputs.append([1,i])
            targets.append(2*i)
        
        dataSet.inputs = np.matrix(inputs)
        dataSet.targets = np.matrix(targets).T
        self.dataSet = dataSet

    def tearDown(self):
        pass


    def test_constructor(self):
        paramSize = 5
        linreg = LinearRegression(paramSize)
        assert linreg.nrParams == paramSize 
        
    def test_trainExecutes(self):
        linreg = LinearRegression(2)
        linreg.train(self.dataSet)
        
    def test_predictExecutes(self):
        linreg = LinearRegression(2)
        linreg.train(self.dataSet)
        
        class LocalDataSet(object):
            inputs = [2]
            
        dataSet = LocalDataSet()     
        linreg.predict(dataSet)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()