'''
Created on Mar 18, 2015

@author: xapharius
'''
import unittest
from managers import ModelManager
from sklearn.linear_model import LinearRegression
from datahandler.numerical import NumericalFeatureSelector
import numpy as np

class Test(unittest.TestCase):

    def setUp(self):
        datapath = "../../../../data/wine-quality/winequality-red.csv"
        self.data = np.loadtxt(open(datapath, "rb"), delimiter = ";")

    def test_train(self):
        model = LinearRegression()
        feature_selector = NumericalFeatureSelector(11, 1)
        manager = ModelManager(model, feature_selector)
        manager.train(self.data)
        assert manager.training_performance is not None

    def test_predict(self):
        model = LinearRegression()
        feature_selector = NumericalFeatureSelector(11, 1)
        manager = ModelManager(model, feature_selector)
        manager.train(self.data)
        results = manager.predict(self.data[:10,:])
        assert results.shape[0] == 10

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()