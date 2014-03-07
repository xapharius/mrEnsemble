'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataSet import AbstractDataSet
import numpy as np

class NumericalDataSet(AbstractDataSet):
    '''
    Dataset for numerical data used either for regression or classification.
    Contains fields inputs and labels as numpy.ndarray
    '''


    def __init__(self, inputs, targets):
        '''
        Constructor
        @param inputs: numpy.ndarray (nr_obs * nr_vars)
        @param targets: numpy.ndarray (nr_obs * nr_vars)
        '''
        self.inputs = inputs
        self.targets = targets
        self.nrInputVars = inputs.shape[1]
        self.nrLabelVars = targets.shape[1]
        self.nrObservations = inputs.shape[0]
        
    def get_observation(self, nr):
        '''
        Get Observation from input matrix as tuple of input and target
        @param nr: number of observation to return
        @return: input and target of observation
        @rtype: tuple (2dim-ndarray, 2dim-ndarray) 
        '''
        inputArr = np.array([self.inputs[nr, :]])
        targetArr = np.array([self.targets[nr, :]])
        return inputArr, targetArr
    
    def gen_observations(self):
        '''
        Iterate over all observations using a generator.
        @return: generator of tuple of inputs and labels
        @rtype: tuple (2dim-ndaray, 2dim-ndarray)
        '''
        for i in range(self.nrObservations):
            inputArr, targetArr = self.get_observation(i)
            yield inputArr, targetArr