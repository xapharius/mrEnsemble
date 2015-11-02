'''
Created on Mar 15, 2015

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractFeatureEngineer(object):
    '''
    Object part of a Model acting as a processor of the raw data to a DataSet object
    Allows feature engineering options to the data eg random subset of features 
    '''

    __metaclass__ = ABCMeta

    def __init__(self, params):
        '''
        Constructor
        '''
        pass

    @abstractmethod
    def get_dataset(self, raw_data):
        '''
        @return: Object implementing AbstractDataSet
        '''
        pass