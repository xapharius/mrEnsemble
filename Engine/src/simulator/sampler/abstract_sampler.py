'''
Created on Feb 18, 2015

@author: xapharius
'''

from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import math

class AbstractSampler(object):
    '''
    Abstract Base Class for a sampler, offering histogram collection for sampling visualization.
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        '''
        Constructor
        '''
        self.dataset = None
        self.nr_obs = None
        self.data_hist = None
        self.sample_hists = None
        self.nr_samples = None

    def bind_data(self, dataset):
        '''
        Bind dataset to Sampler in order to record statistics during sampling
        '''
        self.dataset = dataset
        self.nr_obs = len(dataset)
        self.nr_samples = 0
        self.data_hist = np.zeros(self.nr_obs)   # how often each element has been sampled
        self.sample_hists = []              # array of histograms for each sample

    @abstractmethod
    def sample(self):
        '''
        The actual sampling function. Sampling parameters should be passed to constructor.
        '''
        pass

    def add_sample_histogram(self, sample_hist_arr):
        '''
        Add histogram for sample to the pool of sample histograms
        '''
        assert len(sample_hist_arr) == self.nr_obs, "Sample histogram doesn't contain all obs from the dataset"
        self.sample_hists.append(sample_hist_arr)
        self.data_hist += sample_hist_arr
        self.nr_samples += 1

    def plot_data_histogram(self):
        '''
        Plot histogram showing the total sampled data
        '''
        plt.plot(range(0,self.nr_obs), self.data_hist)
        plt.show()

    def plot_sample_histogram(self, sample_number):
        '''
        Samples generated are indexed starting from 0
        '''
        plt.plot(range(0,self.nr_obs), self.sample_hists[sample_number])
        plt.show()

    def plot_sample_histograms(self):
        '''
        Plot all sample histograms on different subplots
        Cracks for nr_samples < 3 (since subplot is then one dimensional), so return None
        '''
        if self.nr_samples < 3: return
        hEdgeSubplot= int(math.ceil(math.sqrt(self.nr_samples)))
        vEdgeSubplot= int(round(math.sqrt(self.nr_samples)))
        fig, axarr = plt.subplots(vEdgeSubplot, hEdgeSubplot, sharex = True, sharey = True)
        for i in range(self.nr_samples):
            ix = i/hEdgeSubplot
            iy = i%hEdgeSubplot
            axarr[ix, iy].plot(range(self.nr_obs), self.sample_hists[i])
            axarr[ix,iy].axes.get_xaxis().set_visible(False)
            axarr[ix,iy].axes.get_yaxis().set_visible(False)
        #delete empty subplots
        for i in range(self.nr_samples, vEdgeSubplot*hEdgeSubplot):
            ix = i/hEdgeSubplot
            iy = i%hEdgeSubplot
            fig.delaxes(axarr[ix,iy])
        plt.show()
