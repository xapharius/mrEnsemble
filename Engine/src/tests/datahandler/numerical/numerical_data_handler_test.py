'''
Created on Mar 16, 2015

@author: xapharius
'''
import unittest
from datahandler.numerical2.numerical_feature_engineer import NumericalFeatureEngineer
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from datahandler.numerical2.numerical_data_handler import NumericalDataHandler
import numpy as np

class NumericalDatahandlerTest(unittest.TestCase):

    def test_no_rsof(self):
        '''
        No random subset of features, feature selector returns all input features
        '''
        ndh = NumericalDataHandler(random_subset_of_features = False)
        nfs = ndh.get_feature_engineer()
        assert nfs.random_subset_of_features_ratio == 1

    def test_repeated_rsof(self):
        '''
        random subset of features, feature selector returns different number of features "every time" (is random)
        '''
        ndh = NumericalDataHandler(random_subset_of_features = True)
        nfs_nr_feat = []
        for _ in range(10):
            nfs = ndh.get_feature_engineer()
            nfs.get_dataset(np.array([range(100000)]))
            nfs_nr_feat.append(nfs.number_of_features)
        assert len(nfs_nr_feat) == len(set(nfs_nr_feat))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()