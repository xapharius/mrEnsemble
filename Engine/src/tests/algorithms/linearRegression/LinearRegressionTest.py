'''
Created on Dec 4, 2013

@author: xapharius
'''
import unittest
import numpy as np
from algorithms.linearRegression.LinearRegression import LinearRegression
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from numpy.ma.testutils import assert_array_equal
import sys


class LinearRegressionTest(unittest.TestCase):


    def setUp(self):
        csv_source = "../../../../../data/wine-quality/winequality-red.csv"
        rawData = np.genfromtxt(csv_source, delimiter=';', skip_header=0)
        inputs = rawData[:,:11]
        targets = rawData[:,11:]
        
        self.dataSet = NumericalDataSet(inputs, targets)


    def tearDown(self):
        pass


    def test_constructor(self):
        linreg = LinearRegression(self.dataSet.nrInputVars)
        assert linreg.nrParams == 12
        
    def test_addOnes(self):
        linreg = LinearRegression(self.dataSet.nrInputVars)
        matrix = self.dataSet.inputs;
        with1 = linreg.addOnes(matrix)
        assert_array_equal(with1[:,0], np.ones([matrix.shape[0], ]))
        assert_array_equal(with1[:,1:], matrix)
        
    def test_trainExecutes(self):
        linreg = LinearRegression(self.dataSet.nrInputVars)
        linreg.train(self.dataSet)
        
    def test_predictExecutes(self):
        linreg = LinearRegression(self.dataSet.nrInputVars)
        predictions = linreg.predict(self.dataSet)
        assert len(predictions) == self.dataSet.nrObservations
        
    def test_set_params_positive(self):
        nrInputVars = 3;
        linreg = LinearRegression(nrInputVars)
        try:
            linreg.set_params(np.array([[1,2,3,4]]))
        except:
            self.fail("overriding params should be fine");
    
    def test_set_params_negative(self):
        nrInputVars = 3;
        linreg = LinearRegression(nrInputVars)
        try:
            linreg.set_params(np.array([[1,2,3]]))
            self.fail("no exception thrown");
        except:
            expected_errmsg = "overwriting parameters have not the same shape as the model"
            errmsg = str(sys.exc_info()[1])
            assert(errmsg.startswith(expected_errmsg))
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()