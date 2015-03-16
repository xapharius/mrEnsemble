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
        '''
        Test default values after object creation
        '''
        nfs = NumericalFeatureSelector(nr_input_dim = 4)
        assert nfs.nr_input_dim == 4 and nfs.nr_target_dim is None and nfs.random_subset_of_features_ratio == 1
        assert nfs.feature_indices == [0,1,2,3] and nfs.number_of_features == 4

    def test_contructor_exception(self):
        '''
        Test invalid constructor parameters - random_subset_of_features_ratio
        '''
        self.assertRaises(Exception, lambda: NumericalFeatureSelector(nr_input_dim = 4, nr_target_dim = 1, random_subset_of_features_ratio = 3))
        self.assertRaises(Exception, lambda: NumericalFeatureSelector(nr_input_dim = 4, nr_target_dim = 1, random_subset_of_features_ratio = 0))

    def test_full_dataset(self):
        '''
        Test returning all features
        '''
        nfs = NumericalFeatureSelector(nr_input_dim = 2)
        raw_data = np.array([[1,2],[3,4],[5,6]])
        dataset = nfs.get_dataset(raw_data)
        assert type(dataset) is NumericalDataSet
        assert dataset.inputs.shape == raw_data.shape 

    def test_subset_of_features(self):
        '''
        Test (first) run of nfs
        ''' 
        nfs = NumericalFeatureSelector(nr_input_dim = 2, nr_target_dim = 1, random_subset_of_features_ratio= 0.5)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        dataset = nfs.get_dataset(raw_data)
        assert dataset.inputs.shape == (3,1)
        assert dataset.targets.shape == (3,1)
        assert dataset.inputs[0][0] == 1 or dataset.inputs[0][0] == 2

    def test_repeated_call(self):
        '''
        See if nfs setting stay the same after repeated calls
        '''
        nfs = NumericalFeatureSelector(nr_input_dim = 2, nr_target_dim = 1, random_subset_of_features_ratio= 0.5)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        dataset = nfs.get_dataset(raw_data)
        assert dataset.inputs.shape == (3,1)
        assert dataset.targets.shape == (3,1)
        features = nfs.feature_indices

        dataset = nfs.get_dataset(raw_data)
        assert dataset.inputs.shape == (3,1)
        assert dataset.targets.shape == (3,1)
        print (features == nfs.feature_indices)
        assert (features == nfs.feature_indices)

    def test_alteast_one_feature(self):
        '''
        Returns atleast one feature
        '''
        nfs = NumericalFeatureSelector(nr_input_dim = 2, nr_target_dim = 1, random_subset_of_features_ratio= 0.001)
        assert nfs.number_of_features == 1

    def test_mismatching_dimensions(self):
        '''
        Test if Exception is thrown if input and target dimensions dont cover all dataset columns
        '''
        nfs = NumericalFeatureSelector(nr_input_dim = 1, nr_target_dim = 1, random_subset_of_features_ratio= 0.5)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        self.assertRaises(Exception, lambda: nfs.get_dataset(raw_data))
        nfs = NumericalFeatureSelector(nr_input_dim = 1, nr_target_dim = 4, random_subset_of_features_ratio= 0.5)
        self.assertRaises(Exception, lambda: nfs.get_dataset(raw_data))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()