'''
Created on Feb 18, 2015

@author: xapharius
'''

from abstract_sampler import AbstractSampler
import numpy as np
import random

class BootstrapSampler(AbstractSampler):
    '''
    Samples with replacement from given data at a size_ratio specified in the constructor
    '''

    def __init__(self, sample_size_ratio = None):
        '''
        Constructor
        @param sample_ratio: size of the bootstrap sample as a percentage of the dataset's size [0,1].
        Default value is to sample the same nr as the dataset's size (1)
        '''
        if sample_size_ratio is not None and (sample_size_ratio < 0 or sample_size_ratio > 1):
            raise Exception("sample_size_ratio not between 0 and 1")
        self.sample_size_ratio = sample_size_ratio
        self.sample_size = None

    def bind_data(self, dataset):
        super(type(self), self).bind_data(dataset)
        if self.sample_size_ratio is None: 
            self.sample_size_ratio = 1

        self.sample_size = int(round(self.sample_size_ratio * self.nr_obs))

    def sample(self):
        '''
        Create bootstrap sample
        @return: ndarray
        '''
        if self.dataset is None:
            raise Exception("Data not bound")

        sample_hist = self.nr_obs * [0]
        sample_data = []   # emtpy list, vstack later

        for _ in range(self.sample_size):
            index = random.randint(0, self.nr_obs-1)
            sample_data.append(self.dataset[index])
            sample_hist[index] += 1

        sample_data = np.vstack(sample_data)
        self.add_sample_histogram(sample_hist)

        return sample_data