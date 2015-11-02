'''
Created on Feb 18, 2015

@author: xapharius
'''
import os
import unittest
import numpy as np

from simulation.sampler.bootstrap_sampler import BootstrapSampler
import simulation.benchmarker.dataset_loader as dloader

class BootstrapSamplerTest(unittest.TestCase):


    def setUp(self):
        dir_path =  os.getcwd().split("Engine")[0]
        datapath = dir_path + "data/wine-quality/winequality-red.csv"
        data = np.loadtxt(open(datapath, "rb"), delimiter = ";")
        self.inputs = data[:,:-1]
        self.targets = data[:,-1:]

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
        bs = BootstrapSampler(sample_size_ratio=1)
        bs.bind_data(self.inputs)
        assert bs.sample_size == len(self.inputs)

    def test_sample(self):
        bs = BootstrapSampler(0.5)
        bs.bind_data(self.inputs, self.targets)
        sample_inputs, sample_targets = bs.sample()

        assert bs.nr_samples == 1
        assert len(bs.sample_hists) == 1
        assert sample_inputs.shape[0] == int(round(0.5 * len(self.inputs)))
        assert sample_inputs.shape[1] == self.inputs.shape[1]
        assert sample_targets.shape[1] == self.targets.shape[1]
        assert not (bs.data_hist == np.zeros(len(self.inputs))).all()

    def test_without_replacement_even(self):
        bs = BootstrapSampler(0.5, with_replacement=False)
        data = np.arange(10).reshape((10,1))
        bs.bind_data(data)
        sample1_inputs, _ = bs.sample()
        sample2_inputs, _ = bs.sample()
        assert (set(sample1_inputs.ravel()) - set(sample2_inputs.ravel())) == set(sample1_inputs.ravel())
        assert len(sample1_inputs) == len(sample2_inputs)

    def test_without_replacement_uneven(self):
        '''
        1 observation left over
        '''
        bs = BootstrapSampler(0.33, with_replacement=False)
        data = np.arange(10).reshape((10,1))
        bs.bind_data(data)
        bs.sample()
        bs.sample()
        bs.sample()
        sample_inputs, _ = bs.sample()
        assert len(sample_inputs) == 1

    def test_without_replacement_exception(self):
        '''
        Running out of examples, since without replacement
        '''
        bs = BootstrapSampler(0.5, with_replacement=False)
        data = np.arange(10).reshape((10,1))
        bs.bind_data(data)
        _ = bs.sample()
        _ = bs.sample()
        self.assertRaises(Exception, bs.sample)

    def test_image_sampling(self):
        rawdataset = dloader._get_mnist(100)
        bs = BootstrapSampler(0.5, with_replacement=False)
        bs.bind_data(rawdataset.training_inputs, rawdataset.training_targets)
        sin, sout = bs.sample()
        assert len(sin) == 50
        assert len(sout) == 50
        assert sin[0].shape == (28, 28)
        assert sout[0].shape == (10,)
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