'''
Created on Mar 15, 2015

@author: xapharius
'''
import unittest
from datahandler.numerical.numerical_feature_selector import NumericalFeatureSelector
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import numpy as np

class Test(unittest.TestCase):


    def test_constructor(self):
        nfs = NumericalFeatureSelector(nr_input_dim = 4)
        assert nfs.nr_input_dim == 4 and nfs.nr_target_dim is None and nfs.random_subset_of_features_ratio == 1
        assert nfs.feature_indices is None and nfs.number_of_features is None

    def test_contructor_exception(self):
        self.assertRaises(Exception, lambda: NumericalFeatureSelector(nr_input_dim = 4, nr_target_dim = 1, random_subset_of_features_ratio = 3))

    def test_full_dataset(self):
        nfs = NumericalFeatureSelector(nr_input_dim = 2)
        raw_data = np.array([[1,2],[3,4],[5,6]])
        dataset = nfs.get_dataset(raw_data)
        assert type(dataset) is NumericalDataSet
        assert dataset.inputs.shape == raw_data.shape 

    def test_subset_of_features(self): 
        nfs = NumericalFeatureSelector(nr_input_dim = 2, nr_target_dim = 1, random_subset_of_features_ratio= 0.5)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        dataset = nfs.get_dataset(raw_data)
        assert dataset.inputs.shape == (3,1)
        assert dataset.targets.shape == (3,1)
        assert dataset.inputs[0][0] == 1 or dataset.inputs[0][0] == 2

    def test_mismatching_dimensions(self):
        nfs = NumericalFeatureSelector(nr_input_dim = 1, nr_target_dim = 1, random_subset_of_features_ratio= 0.5)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        self.assertRaises(Exception, lambda: nfs.get_dataset(raw_data))
        nfs = NumericalFeatureSelector(nr_input_dim = 1, nr_target_dim = 4, random_subset_of_features_ratio= 0.5)
        self.assertRaises(Exception, lambda: nfs.get_dataset(raw_data))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()