'''
Created on Mar 15, 2015

@author: xapharius
'''

from datahandler.abstract_feature_selector import AbstractFeatureSelector
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import random

class NumericalFeatureSelector(AbstractFeatureSelector):
    '''
    Selecting features for the model it was assigned to, allowing individualization of models
    '''

    def __init__(self, nr_input_dim, nr_target_dim = None, random_subset_of_features_ratio = 1):
        '''
        @param random_subset_of_features_ratio: float (0,1] representing % of features
        '''
        if not(random_subset_of_features_ratio > 0 and random_subset_of_features_ratio <= 1):
            raise Exception("random_subset_of_features_ratio not in (0,1]")
        self.nr_input_dim = nr_input_dim
        self.nr_target_dim = nr_target_dim
        self.random_subset_of_features_ratio = random_subset_of_features_ratio
        # select features
        if self.random_subset_of_features_ratio == 1:
            self.number_of_features = nr_input_dim
            self.feature_indices = range(nr_input_dim)
        else:
            # create index vector for random subset of features
            self.number_of_features = int(round(self.random_subset_of_features_ratio * nr_input_dim))
            if self.number_of_features == 0: self.number_of_features = 1 # take atleast one feature
            self.feature_indices = random.sample(range(nr_input_dim), self.number_of_features)
        super(AbstractFeatureSelector, self).__init__()

    def get_dataset(self, raw_data):
        '''
        @param raw_data: numpy.ndarray inputs and targets
        '''
        targets = None #default, in case not supplied

        if raw_data.shape[1] > self.nr_input_dim:
            # should contain targets as well
            if self.nr_target_dim is not None:
                if self.nr_input_dim + self.nr_target_dim != raw_data.shape[1]:
                    raise Exception("input and target dimensions don't add up to raw_data column size")
                targets = raw_data[:, raw_data.shape[1] - self.nr_target_dim:]
        elif raw_data.shape[1] < self.nr_input_dim:
            raise Exception("Too few variables")

        # else = only inputs are supplied
        inputs = raw_data[:, :self.nr_input_dim]
        input_subset = inputs[:, self.feature_indices]

        #TODO: maybe generate some polinomial or pca features here

        return NumericalDataSet(input_subset, targets)


