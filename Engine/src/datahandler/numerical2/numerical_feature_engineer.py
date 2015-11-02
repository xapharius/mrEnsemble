'''
Created on Mar 15, 2015

@author: xapharius
'''

from datahandler.abstract_feature_engineer import AbstractFeatureEngineer
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import random

class NumericalFeatureEngineer(AbstractFeatureEngineer):
    '''
    Selecting features for the model it was assigned to, allowing individualization of models
    Selection is done when get_dataset is called 
    '''

    def __init__(self, random_subset_of_features_ratio=1):
        '''
        @param random_subset_of_features_ratio: float (0,1] representing % of features
        '''
        if not(random_subset_of_features_ratio > 0 and random_subset_of_features_ratio <= 1):
            raise Exception("random_subset_of_features_ratio not in (0,1]")
        self.random_subset_of_features_ratio = random_subset_of_features_ratio
        self.number_of_features = None
        self.feature_indices = None

    def _init_selection(self, nr_input_dim):
        '''
        Randomly select some features based on ratio param
        '''
        if self.random_subset_of_features_ratio == 1:
            self.number_of_features = nr_input_dim
            self.feature_indices = range(nr_input_dim)
        else:
            # create index vector for random subset of features
            self.number_of_features = int(round(self.random_subset_of_features_ratio * nr_input_dim))
            if self.number_of_features == 0: self.number_of_features = 1 # take atleast one feature
            self.feature_indices = random.sample(range(nr_input_dim), self.number_of_features)

    def get_dataset(self, raw_inputs, targets=None):
        '''
        @param raw_inputs: numpy.ndarray only inputs
        @param targets: numpy.ndarray (n,1)
        '''
        if self.number_of_features is None:
            # not initialized - choose features first
            self._init_selection(raw_inputs.shape[1])

        if raw_inputs.shape[1] < max(self.feature_indices)+1:
            raise Exception("Too few variables")

        inputs = raw_inputs[:, self.feature_indices]
        if targets is not None and len(targets.shape) == 2 and targets.shape[1] == 1:
            targets = targets.ravel()
        #TODO: maybe generate some polynomial features here

        return NumericalDataSet(inputs, targets)


