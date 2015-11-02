'''
Created on Mar 15, 2015

@author: xapharius
'''

from datahandler.numerical2.numerical_feature_engineer import NumericalFeatureEngineer
import random

class NumericalDataHandler(object):
    '''
    Factory for NumericalFeatureEngineers
    '''


    def __init__(self, random_subset_of_features = False):
        '''
        Factory Settings
        @param random_subset_of_features: Boolean whether to use method. Percentage of features to use is drawn randomly (0, 1).
        '''
        self.random_subset_of_features = random_subset_of_features

    def get_feature_engineer(self):
        random_percentage_of_featues = random.random() if self.random_subset_of_features is True else 1
        return NumericalFeatureEngineer(random_percentage_of_featues)