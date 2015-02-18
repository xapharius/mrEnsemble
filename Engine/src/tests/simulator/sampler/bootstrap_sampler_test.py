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
        bs = BootstrapSampler(100)
        assert bs.sampleSize == 100

    def test_data_not_bound(self):
        bs = BootstrapSampler(100)
        try:
            bs.sample()
        except Exception:
            return
        assert False

    def test_sampleSize(self):
        bs = BootstrapSampler()
        assert bs.sampleSize == None
        bs.bind_data(self.data)
        assert bs.sampleSize == self.data.shape[0]

    def test_sample(self):
        bs = BootstrapSampler(100)
        bs.bind_data(self.data)
        sample = bs.sample()

        assert bs.nrSamples == 1
        assert len(bs.sample_hists) == 1
        assert sample.shape[0] == 100
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