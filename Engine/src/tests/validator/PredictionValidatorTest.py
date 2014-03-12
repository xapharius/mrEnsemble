'''
Created on Mar 12, 2014

@author: xapharius
'''
import unittest
import numpy as np
from validator.PredictionValidator import PredictionValidator
from numpy.ma.testutils import assert_equal
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from algorithms.linearRegression import LinearRegression


class Test(unittest.TestCase):


    def test_computeMSE(self):
        targets = [np.array([[-2]]), np.array([[2]]), np.array([[4]])]
        predictions = [np.array([[1]]), np.array([[-1]]), np.array([[4]])]
        val = PredictionValidator()
        mse = val.computeMSE(targets, predictions)
        # cumputed per hand
        assert_equal(round(mse,3), 6, "wrong mse")
        
    def test_computeR2(self):
        targets = [np.array([[0]]), np.array([[2]]), np.array([[4]]), np.array([[6]])]
        predictions = [np.array([[0]]), np.array([[1.5]]), np.array([[3]]), np.array([[4.5]])]
        val = PredictionValidator()
        r2 = val.computeR2(targets, predictions)
        # computed per hand ssr/sst
        # I HAVE NO IDEA WHY THE EQUATIONS ARE NOT EQUAL
        assert_equal(round(r2,3), 0.675, "wrong R-Squared")
    
    def test_validate_local(self):
        csv_source = "../../../../data/wine-quality/winequality-red.csv"
        rawData = np.genfromtxt(csv_source, delimiter=';', skip_header=0)
        inputs = rawData[:,:11]
        targets = rawData[:,11:]
        dataSet = NumericalDataSet(inputs, targets)
        
        linreg = LinearRegression(dataSet.nrInputVars)
        predictions = linreg.predict(dataSet)
        
        val = PredictionValidator()
        val.validate_local(linreg, dataSet)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_computeMSE']
    unittest.main()