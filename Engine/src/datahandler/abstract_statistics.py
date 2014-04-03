'''
Created on Mar 26, 2014

@author: Simon
'''

from abc import ABCMeta, abstractmethod

class AbstractStatistics(object):
    '''
    Base class for statistics.
    '''

    __metaclass__ = ABCMeta

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self, encoded_stats):
        pass