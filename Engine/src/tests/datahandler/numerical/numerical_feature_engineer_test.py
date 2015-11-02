'''
Created on Mar 15, 2015

@author: xapharius
'''
import unittest
from datahandler.numerical2.numerical_feature_engineer import NumericalFeatureEngineer
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import numpy as np

class NumericalFeatureEngineerTest(unittest.TestCase):


    def test_constructor(self):
        '''
        Test default values after object creation
        '''
        nfs = NumericalFeatureEngineer()
        assert nfs.random_subset_of_features_ratio == 1
        assert nfs.feature_indices is None and nfs.number_of_features is None

    def test_contructor_exception(self):
        '''
        Test invalid constructor parameters - random_subset_of_features_ratio
        '''
        self.assertRaises(Exception, lambda: NumericalFeatureEngineer(random_subset_of_features_ratio = 3))
        self.assertRaises(Exception, lambda: NumericalFeatureEngineer(random_subset_of_features_ratio = 0))

    def test_full_dataset(self):
        '''
        Test returning all features
        '''
        nfs = NumericalFeatureEngineer()
        raw_data = np.array([[1, 2, 3],[3, 4, 5],[5, 6, 7]])
        dataset = nfs.get_dataset(raw_data[:,:-1], raw_data[:,-1:])
        assert type(dataset) is NumericalDataSet
        assert dataset.inputs.shape == (raw_data.shape[0], raw_data.shape[1]-1)
        assert dataset.targets.shape == (raw_data.shape[0], 1) 

    def test_subset_of_features(self):
        '''
        Test (first) run of nfs
        ''' 
        nfs = NumericalFeatureEngineer(random_subset_of_features_ratio=0.5)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        dataset = nfs.get_dataset(raw_data[:,:-1], raw_data[:,-1:])
        assert dataset.inputs.shape == (3,1)
        assert dataset.targets.shape == (3,1)
        assert dataset.inputs[0][0] == 1 or dataset.inputs[0][0] == 2

    def test_repeated_call(self):
        '''
        See if nfs setting stay the same after repeated calls
        '''
        nfs = NumericalFeatureEngineer(random_subset_of_features_ratio=0.5)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        dataset = nfs.get_dataset(raw_data[:,:-1], raw_data[:,-1:])
        assert dataset.inputs.shape == (3,1)
        assert dataset.targets.shape == (3,1)
        features = nfs.feature_indices

        raw_data = np.array([[1,2,5],[3,4,5],[5,6,5]])
        dataset = nfs.get_dataset(raw_data[:,:-1], raw_data[:,-1:])
        assert dataset.inputs.shape == (3,1)
        assert dataset.targets.shape == (3,1)
        assert (features == nfs.feature_indices)

    def test_alteast_one_feature(self):
        '''
        Returns atleast one feature
        '''
        nfs = NumericalFeatureEngineer(random_subset_of_features_ratio=0.001)
        nfs.get_dataset(np.array([[1,2,5],[3,4,5],[5,6,5]]))
        assert nfs.number_of_features == 1

    def test_mismatching_dimensions(self):
        '''
        Test if Exception is thrown if input and target dimensions dont cover all dataset columns
        '''
        nfs = NumericalFeatureEngineer(random_subset_of_features_ratio=1)
        raw_data = np.array([[1,2,9],[3,4,9],[5,6,9]])
        nfs.get_dataset(raw_data)
        raw_data = np.array([[1,9],[3,9],[5,9]])
        self.assertRaises(Exception, lambda: nfs.get_dataset(raw_data))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()