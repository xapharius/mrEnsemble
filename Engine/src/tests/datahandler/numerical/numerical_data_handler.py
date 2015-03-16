'''
Created on Mar 16, 2015

@author: xapharius
'''
import unittest
from datahandler.numerical.numerical_feature_selector import NumericalFeatureSelector
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from datahandler.numerical.numerical_data_handler import NumericalDataHandler
import numpy as np

class Test(unittest.TestCase):

    def test_no_rsof(self):
        '''
        No random subset of features, feature selector returns all input features
        '''
        ndh = NumericalDataHandler(nr_input_dim = 4, nr_target_dim = 1, random_subset_of_features = False)
        nfs = ndh.get_feature_selector()
        assert nfs.number_of_features == 4 and nfs.feature_indices == range(4)

    def test_repeated_rsof(self):
        '''
        random subset of features, feature selector returns different number of features "every time" (is random)
        '''
        ndh = NumericalDataHandler(nr_input_dim = 10000, nr_target_dim = 1, random_subset_of_features = True)
        nfs_nr_feat = [ndh.get_feature_selector().number_of_features for _ in range(4)]
        assert len(nfs_nr_feat) == len(set(nfs_nr_feat))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()