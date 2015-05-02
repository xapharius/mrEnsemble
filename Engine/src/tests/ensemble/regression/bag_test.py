'''
Created on Mar 22, 2015

@author: xapharius
'''
import unittest
import numpy as np
from simulator.sampler.bootstrap_sampler import BootstrapSampler
from factory.homogenous_factory import HomogenousFactory
from datahandler.numerical.numerical_data_handler import NumericalDataHandler
from sklearn.linear_model import LinearRegression
from ensemble.regression.bag import Bag
import os

class BagTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path =  os.getcwd().split("Engine")[0]
        datapath = dir_path + "data/wine-quality/winequality-red.csv"
        cls.data = np.loadtxt(open(datapath, "rb"), delimiter = ";")
        cls.sampler = BootstrapSampler(sample_size_ratio = 0.1)
        cls.sampler.bind_data(cls.data)

    def test_run(self):
        datahandler = NumericalDataHandler(nr_input_dim = 11, nr_target_dim = 1, random_subset_of_features=True)
        factory = HomogenousFactory(datahandler = datahandler, algorithm_class= LinearRegression)
        manager1 = factory.get_instance()
        manager1.train(self.sampler.sample())
        manager2 = factory.get_instance()
        manager2.train(self.sampler.sample())
        manager3 = factory.get_instance()
        manager3.train(self.sampler.sample())
        bag = Bag([manager1, manager2, manager3])
        # print bag.predict(self.sampler.sample())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()