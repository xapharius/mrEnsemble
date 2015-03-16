'''
Created on Mar 15, 2015

@author: xapharius
'''

from datahandler.numerical.numerical_feature_selector import NumericalFeatureSelector
import random

class NumericalDataHandler(object):
    '''
    Factory for NumericalFeatureSelectors
    '''


    def __init__(self, nr_input_dim, nr_target_dim = None, random_subset_of_features = False):
        '''
        Factory Settings
        @param random_subset_of_features: Boolean whether to use method. Percentage of features to use is drawn randomly (0, 1).
        '''
        self.nr_input_dim =  nr_input_dim
        self.nr_target_dim =  nr_target_dim
        self.random_subset_of_features = random_subset_of_features

    def get_feature_selector(self):
        random_percentage_of_featues = random.random() if self.random_subset_of_features is True else 1
        return NumericalFeatureSelector(self.nr_input_dim, self.nr_target_dim, random_percentage_of_featues)