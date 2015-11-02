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

    def __init__(self, sample_size_ratio=0.1, with_replacement=True):
        '''
        Constructor
        @param sample_ratio: size of the bootstrap sample as a percentage of the dataset's size [0,1].
        Default value is to sample the same nr as the dataset's size (1)
        '''
        if sample_size_ratio is not None and (sample_size_ratio < 0 or sample_size_ratio > 1):
            raise Exception("sample_size_ratio not between 0 and 1")
        self.sample_size_ratio = sample_size_ratio
        self.sample_size = None
        self.with_replacement = with_replacement
        self.sampled_indices = None

    def bind_data(self, inputs, targets=None):
        '''
        @param inputs: iterabel that has observations on first axis
        '''
        super(type(self), self).bind_data(inputs, targets)
        self.sample_size = int(round(self.sample_size_ratio * self.nr_obs))
        self.sampled_indices = np.zeros(self.nr_obs)


    def sample(self):
        '''
        Create bootstrap sample
        @return: ndarray
        '''
        if self.inputs is None:
            raise Exception("Data not bound")
        if sum(self.sampled_indices) == len(self.sampled_indices):
            raise Exception("No Observations left") 

        sample_hist = self.nr_obs * [0]
        sample_inputs = []   # emtpy list, vstack later
        sample_targets = []

        if len(self.sampled_indices) - sum(self.sampled_indices) < self.sample_size:
            curr_sample_size = int(len(self.sampled_indices) - sum(self.sampled_indices))
        else:
            curr_sample_size = self.sample_size

        for _ in range(curr_sample_size):
            index = random.choice(np.where(self.sampled_indices == 0)[0])
            if not self.with_replacement:
                self.sampled_indices[index] = 1
            sample_inputs.append(self.inputs[index])
            if self.targets is not None:
                sample_targets.append(self.targets[index])
            sample_hist[index] += 1

        sample_inputs = np.array(sample_inputs)
        if self.targets is not None:
            sample_targets = np.array(sample_targets)
        else:
            sample_targets = None
        self.add_sample_histogram(sample_hist)

        return sample_inputs, sample_targets