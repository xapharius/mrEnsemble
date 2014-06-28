'''
Created on Jun 19, 2013

@author: Simon
'''

from algorithms.AbstractAlgorithmFactory import AbstractAlgorithmFactory
from algorithms.image.gaussian_filter import GaussianFilter

class GaussianFilterFactory(AbstractAlgorithmFactory):
    '''
    Factory class for Linear Regression.
    Provides the functionalities specified by the AbstractAlgorithmClass.
    '''


    def __init__(self, sigma):
        '''
        Constructor
        '''
        self.sigma = sigma
    
    def get_instance(self):
        '''
        Create a Gaussian filter
        :return: Object implementing AbstractAlgorithm
        '''
        gaussian_filter = GaussianFilter(self.sigma)
        return gaussian_filter

    def aggregate(self, algs):
        '''
        Aggregates a list of GaussianFilter instances.
        :param algs list of algorithm instances
        :return same as input
        '''
        all_filtered = []
        for f in algs:
            all_filtered.extend(f.filtered)
        gaussian_filter = GaussianFilter(self.sigma)
        gaussian_filter.filtered = all_filtered
        return gaussian_filter
    
    def encode(self, alg_instance):
        return alg_instance.filtered
    
    def decode(self, encoded):
        deserialized = []
        for f in encoded:
            # create new algorithm object
            gauss_filter = GaussianFilter(self.sigma)
            gauss_filter.set_params(f)
            # append to list of algorithm objetcs
            deserialized.append(gauss_filter)
        return deserialized
