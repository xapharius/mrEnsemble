'''
Created on Feb 18, 2015

@author: xapharius
'''
import unittest
from simulator.sampler.abstract_sampler import AbstractSampler
import numpy as np

class concAS(AbstractSampler):
    def sample(self):
        pass

class Test(unittest.TestCase):

 
    def setUp(self):
        self.dataset = np.array([[1,1,1],[2,2,2],[3,3,3]])
        self.asmpl = concAS()
        self.asmpl.bind_data(self.dataset)

    def test_bind_data(self):
        assert self.asmpl.nrObs == 3
        assert self.asmpl.dataset is not None
        assert (self.asmpl.data_hist == [0,0,0]).all()
        assert self.asmpl.nrSamples == 0

    def test_add_sample_histogram(self):
        self.asmpl.add_sample_histogram([1,1,2])
        self.asmpl.add_sample_histogram([1,2,5])
        print self.asmpl.data_hist
        assert (self.asmpl.data_hist == [2,3,7]).all()
        assert len(self.asmpl.sample_hists) == 2
        assert self.asmpl.nrSamples == 2

    '''
    #Visual Test
    def test_plot_data_histogram(self):
        self.asmpl.data_hist = np.array([1,2,3])
        self.asmpl.plot_data_histogram()
    '''

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()