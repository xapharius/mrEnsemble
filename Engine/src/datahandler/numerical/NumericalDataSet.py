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


    def __init__(self, inputs, targets):
        '''
        Constructor
        @param inputs: numpy.ndarray
        @param targets: numpy.ndarray
        '''
        self.inputs = inputs
        self.targets = targets
        self.nrInputVars = inputs.shape[1]
        self.nrLabelVars = targets.shape[1]
        self.nrObservations = inputs.shape[0]
        