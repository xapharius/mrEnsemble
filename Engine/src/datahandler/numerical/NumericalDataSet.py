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


    def __init__(self, inputs, targets=None):
        '''
        Constructor
        @param inputs: 2-dim numpy.ndarray (nr_obs * nr_vars)
        @param targets: 2-dim numpy.ndarray (nr_obs * nr_vars)
        @raise exception: if targets not null then they must have same nr of observations 
        '''
        self.inputs = inputs
        self.targets = targets
        self.nrObservations = len(inputs)

        if targets is not None:
            self.nrTargetVars = targets.shape[1] if len(targets.shape) == 2 else 1
            if len(inputs) != len(targets):
                raise Exception("number of inputs and targets observations mismatch")
        else:
            self.nrTargetVars = 0

    def get_observation(self, nr):
        '''
        Get Observation from input matrix as tuple of input and target
        @param nr: number of observation to return
        @return: input and target of observation
        @rtype: tuple (2dim-ndarray, 2dim-ndarray) 
        '''
        inputArr = np.array([self.inputs[nr]])
        if self.targets is not None:
            targetArr = np.array([self.targets[nr, :]])
        else:
            targetArr = None
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

    def rand_observation(self):
        i = np.random.randint(0, high=self.nrObservations)
        return self.get_observation(i)

    def get_inputs(self):
        return self.inputs

    def get_targets(self):
        return self.targets

    def get_nr_observations(self):
        return self.nrObservations
