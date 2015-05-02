'''
Created on Mar 23, 2015

@author: xapharius
'''
import unittest
import numpy as np
from datahandler.numerical.numerical_data_handler import NumericalDataHandler
from factory import HomogenousFactory
from sklearn.linear_model import LinearRegression
from datahandler.numerical.numerical_feature_selector import NumericalFeatureSelector
import os

class HomogenousFactoryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path =  os.getcwd().split("Engine")[0]
        datapath = dir_path + "data/wine-quality/winequality-red.csv"
        cls.data = np.loadtxt(open(datapath, "rb"), delimiter = ";")
        cls.datahandler = NumericalDataHandler(11, 1, random_subset_of_features = True)

    def test_constructor(self):
        try:
            HomogenousFactory(self.datahandler, LinearRegression)
        except:
            assert False

    def test_get_instance(self):
        factory = HomogenousFactory(self.datahandler, LinearRegression)
        manager1 = factory.get_instance()
        assert isinstance(manager1.model, LinearRegression)
        assert isinstance(manager1.feature_selector, NumericalFeatureSelector)
        assert manager1.feature_selector.number_of_features in range(1,11)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()