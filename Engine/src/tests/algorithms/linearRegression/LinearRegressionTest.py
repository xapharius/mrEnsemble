'''
Created on Dec 4, 2013

@author: xapharius
'''
import unittest
import numpy as np
from algorithms.linearRegression.LinearRegression import LinearRegression


class LinearRegressionTest(unittest.TestCase):


    def setUp(self):
        class localDataSet(object):
            pass  
        dataSet = localDataSet()
        
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


    def testConstructor(self):
        paramSize = 5
        linreg = LinearRegression(paramSize)
        assert linreg.params.size == paramSize 
        
    def testTrainExecutes(self):
        linreg = LinearRegression(2)
        linreg.train(self.dataSet)
        
    def testPredictExecutes(self):
        linreg = LinearRegression(2)
        linreg.train(self.dataSet)
        
        class localDataSet(object):
            inputs = [2]
            
        dataSet = localDataSet()     
        linreg.predict(dataSet)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()