'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler import AbstractDataSet

class NumericalDataSet(AbstractDataSet):
    '''
    Dataset for numerical data used either for regression or classification.
    Contains fields inputs and labels as numpy.ndarray
    '''


    def __init__(self, inputs, labels):
        '''
        Constructor
        @param inputs: numpy.ndarray
        @param labels: numpy.ndarray
        '''
        self.inputs = inputs
        self.labels = labels
        