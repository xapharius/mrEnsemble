'''
Created on Dec 4, 2013

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
from skimage import filter
import sys

class GaussianFilter(AbstractAlgorithm):
    '''
    classdocs
    '''


    def __init__(self, sigma):
        '''
        Constructor
        '''
        self.sigma = sigma
        self.filtered = []

    def train(self, _dataSet):
        '''
        Trains Model for given dataset (ImageDataSet)
        '''
        sys.stderr.write("Got " + str(len(_dataSet.get_inputs())) + " images\n")
        for im in _dataSet.get_inputs():
            self.filtered.append(filter.gaussian_filter(im, self.sigma, multichannel=False))

    def predict(self, _dataSet):
        '''
        Predicts targets for given dataset.inputs
        @return: predictions
        @rtype: list of np.arrays
        '''
        pass


    def set_params(self, filtered):
        '''Set list of filtered images.
        @param filtered: list of np.array
        '''
        self.filtered = filtered
