'''
Created on Jan 11, 2014

@author: xapharius
'''
import unittest
from datahandler.numerical.NumericalDataProcessor import NumericalDataProcessor 
from numpy.ma.testutils import assert_equal, assert_array_almost_equal
import numpy as np
from datahandler.numerical.numerical_pre_processor import NumericalPreProcessor
from datahandler.numerical.numerical_stats import NumericalStats


class NumericalDataProcessorTest(unittest.TestCase):

    def setUp(self):
        self.rawData = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6]])


    def test_constructor(self):
        data_proc = NumericalDataProcessor(3, 1)
        data_proc.set_data(self.rawData)
        data_set = data_proc.get_data_set()
        assert_equal(data_set.inputs.shape, (3,3))
        assert_equal(data_set.targets.shape, (3,1))

    def test_standardize_inputs(self):
        data_proc = NumericalDataProcessor(3, 1)
        data_proc.input_scalling = NumericalDataProcessor.STANDARDIZE
        data_proc.set_data(self.rawData)
        data_set = data_proc.get_data_set()
        stats = NumericalStats().decode(NumericalPreProcessor().calculate(data_set))
        data_proc.normalize_data(stats)
        data_set = data_proc.get_data_set()
        
        inputs = self.rawData[:, :3]
        inputs = (inputs - np.mean(inputs, axis=0)) / np.var(inputs, axis=0)
        
        assert_array_almost_equal(data_set.get_inputs(), inputs)

    def test_standardize_targets(self):
        data_proc = NumericalDataProcessor(3, 1)
        data_proc.target_scalling = NumericalDataProcessor.STANDARDIZE
        data_proc.set_data(self.rawData)
        data_set = data_proc.get_data_set()
        stats = NumericalStats().decode(NumericalPreProcessor().calculate(data_set))
        data_proc.normalize_data(stats)
        data_set = data_proc.get_data_set()
        
        targets = self.rawData[:, 3:]
        targets = (targets - np.mean(targets, axis=0)) / np.var(targets, axis=0)
        
        assert_array_almost_equal(data_set.get_targets(), targets)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()