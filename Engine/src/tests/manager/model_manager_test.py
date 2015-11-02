'''
Created on Mar 18, 2015

@author: xapharius
'''
import unittest
from manager import *
from sklearn.linear_model import LinearRegression
from datahandler.numerical2.numerical_feature_engineer import NumericalFeatureEngineer
import numpy as np
import os

class ModelManagerTest(unittest.TestCase):


    def setUp(self):
        dir_path =  os.getcwd().split("Engine")[0]
        datapath = dir_path + "data/wine-quality/winequality-red.csv"
        self.data = np.loadtxt(open(datapath, "rb"), delimiter=";")


    def test_train(self):
        model = LinearRegression()
        feature_engineer = NumericalFeatureEngineer()
        manager = ModelManager(model, feature_engineer)
        manager.train(self.data)
        assert manager.training_performance is not None


    def test_predict(self):
        model = LinearRegression()
        feature_engineer = NumericalFeatureEngineer()
        manager = ModelManager(model, feature_engineer)
        manager.train(self.data)
        results = manager.predict(self.data[:10,:])
        assert results.shape[0] == 10


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()