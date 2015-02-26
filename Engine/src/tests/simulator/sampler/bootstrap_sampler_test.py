'''
Created on Feb 18, 2015

@author: xapharius
'''
import unittest
import numpy as np
from simulator.sampler.bootstrap_sampler import BootstrapSampler

class Test(unittest.TestCase):


    def setUp(self):
        datapath = "../../../../../data/wine-quality/winequality-red.csv"
        self.data = np.loadtxt(open(datapath, "rb"), delimiter = ";")

    def tearDown(self):
        pass


    def test_constructor(self):
        bs = BootstrapSampler(0.5)
        assert bs.sample_size_ratio == 0.5
        self.assertRaises(Exception, BootstrapSampler, -0.1)
        self.assertRaises(Exception, BootstrapSampler, 10)

    def test_data_not_bound(self):
        bs = BootstrapSampler()
        self.assertRaises(Exception, bs.sample)

    def test_sample_size(self):
        bs = BootstrapSampler()
        assert bs.sample_size == None
        bs.bind_data(self.data)
        assert bs.sample_size == self.data.shape[0]

    def test_sample(self):
        bs = BootstrapSampler(0.5)
        bs.bind_data(self.data)
        sample = bs.sample()

        assert bs.nr_samples == 1
        assert len(bs.sample_hists) == 1
        assert sample.shape[0] == int(round(0.5 * self.data.shape[0]))
        assert sample.shape[1] == self.data.shape[1]
        assert not (bs.data_hist == np.zeros(self.data.shape[0])).all()
    '''
    #Visual test
    def test_plot_histogram(self):
        bs = BootstrapSampler(100)
        bs.bind_data(self.data)
        for _ in range(10):
            bs.sample()
        #bs.plot_data_histogram()
        #bs.plot_sample_histogram(0)
        bs.plot_sample_histograms()
    '''

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()