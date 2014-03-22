'''
Created on Mar 19, 2014

@author: Simon
'''

from abc import ABCMeta, abstractmethod

class AbstractValidator(object):
    '''
    Abstract class for a DataHandler. 
    Specifies which methods a DataHandler should implement. The DataHandler is responsible for a certain Data class
    (e.g images, timeseries etc), so the necessary pre- and dataprocessors belong to the same class and work together. 
    '''
    __metaclass__ = ABCMeta


    @abstractmethod
    def validate(self, alg, data_set):
        '''
        Validates the given algorithm instance using the given data.
        @param alg: Algorithm instance implementing AbstractAlgorithm that 
                    should be validated.
        @param data_set: Instance implementing AbstractDataSet representing the
                         data that should be used for validation.
        @return: Result of validation. Usually some kind of statistics
        '''
        pass

    @abstractmethod
    def aggregate(self, validation_results):
        '''
        Aggregate multiple results returned by 'validate'.
        @return: Final validation result aggregating the given results.
        '''
        pass
