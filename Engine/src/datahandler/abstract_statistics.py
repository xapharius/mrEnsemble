'''
Created on Mar 26, 2014

@author: Simon
'''

from abc import ABCMeta, abstractmethod

class AbstractStatistics(object):
    '''
    Base class for statistics. Actually just a data transfer object for the
    statistics.
    '''

    __metaclass__ = ABCMeta

    @abstractmethod
    def encode(self):
        '''
        Creates a JSON serializable representation of this statistics instance.
        '''
        pass

    @abstractmethod
    def decode(self, encoded_stats):
        '''
        Counterpart to 'encode'. This statistics instance takes over all values
        of the passed encoded instance.
        @param encoded_stats: Encoded statistics instance
        @return: Has to return this instance for convenience. 
        '''
        pass