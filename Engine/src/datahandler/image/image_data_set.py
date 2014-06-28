'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataSet import AbstractDataSet
import numpy as np

class ImageDataSet(AbstractDataSet):
    '''
    Data set for image data.
    Contains the fields inputs and targets as list of numpy.ndarray
    '''


    def __init__(self, inputs, targets = None):
        '''
        Constructor
        @param inputs: list of numpy.ndarray (images)
        @param targets: list of some sort of label
        '''
        self.inputs = inputs
        self.targets = targets
        self.nrObservations = len(inputs)


    def get_observation(self, nr):
        '''
        Get Observation from inputs as tuple of input and target
        @param nr: index of observation to return
        @return: input and target of observation
        @rtype: tuple (numpy.ndarray, target) 
        '''
        inputArr = self.inputs[nr]
        if self.targets != None:
            targetArr = self.targets[nr]
        else:
            targetArr = None
        return inputArr, targetArr

    def gen_observations(self):
        '''
        Iterate over all observations using a generator.
        @return: generator of tuple of inputs and targets
        @rtype: tuple (ndaray, target)
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
