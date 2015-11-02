'''
Created on Mar 23, 2015

@author: xapharius
'''
import unittest
import os
import numpy as np

from sklearn.linear_model import LinearRegression

from datahandler.numerical2.numerical_data_handler import NumericalDataHandler
from factory.algorithm_factory import AlgorithmFactory
from factory import HomogenousFactory

from datahandler.numerical2.numerical_feature_engineer import NumericalFeatureEngineer


class HomogenousFactoryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path =  os.getcwd().split("Engine")[0]
        datapath = dir_path + "data/wine-quality/winequality-red.csv"
        cls.data = np.loadtxt(open(datapath, "rb"), delimiter = ";")
        cls.datahandler = NumericalDataHandler(random_subset_of_features = True)
        cls.alg_factory = AlgorithmFactory(LinearRegression)

    def test_constructor(self):
        try:
            HomogenousFactory(self.datahandler, self.alg_factory)
        except:
            assert False

    def test_get_instance(self):
        factory = HomogenousFactory(self.datahandler, self.alg_factory)
        manager1 = factory.get_instance()
        assert isinstance(manager1.model, LinearRegression)
        assert isinstance(manager1.feature_engineer, NumericalFeatureEngineer)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()