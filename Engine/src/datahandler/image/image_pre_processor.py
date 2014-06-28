'''
Created on Mar 12, 2014

@author: Simon
'''
from datahandler.AbstractPreProcessor import AbstractPreProcessor

class ImagePreProcessor(AbstractPreProcessor):
    '''
    Preprocessor implementation for image data sets.
    Does nothing yet.
    '''

    def calculate(self, data_set):
        return []

    def aggregate(self, key, values):
        return []
