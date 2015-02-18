'''
Created on Feb 18, 2015

@author: xapharius
'''

from abstract_sampler import AbstractSampler
import numpy as np
import random

class BootstrapSampler(AbstractSampler):

    def __init__(self, sampleSize = None):
        '''
        Constructor
        '''
        self.sampleSize = sampleSize

    def bind_data(self, dataset):
        super(type(self), self).bind_data(dataset)
        if self.sampleSize is None: 
            self.sampleSize = self.nrObs

    def sample(self):
        '''
        Create bootstrap sample
        @return: ndarray
        '''
        if self.dataset is None:
            raise Exception("Data not bound")

        sample_hist = self.nrObs * [0]
        sample_data = []   # emtpy list, vstack later

        for _ in range(self.sampleSize):
            index = random.randint(0, self.nrObs-1)
            sample_data.append(self.dataset[index])
            sample_hist[index] += 1

        sample_data = np.vstack(sample_data)
        self.add_sample_histogram(sample_hist)

        return sample_data