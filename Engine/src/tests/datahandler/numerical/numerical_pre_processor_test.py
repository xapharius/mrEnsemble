'''
Created on Mar 17, 2014

@author: linda
'''
import unittest
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from datahandler.numerical.numerical_pre_processor import NumericalPreProcessor
import numpy as np
from numpy.testing.utils import assert_equal, assert_array_almost_equal
from datahandler.numerical.numerical_stats import NumericalStats

class NumericalPreProcessorTest(unittest.TestCase):


    def test_data_stats_calculation(self):
        pre_processor = NumericalPreProcessor()
        data_set = NumericalDataSet(np.random.random_integers(1,10, (10,10)), np.random.random_integers(1,10, (10,1)))
        result = NumericalStats(encoded=pre_processor.calculate(data_set)['data'])
        
        data = data_set.inputs
        size = data.shape[0]
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        var = np.var(data, axis=0)
        
        assert_equal(result.get_size(), size)
        assert_array_almost_equal(result.get_min(), _min)
        assert_array_almost_equal(result.get_max(), _max)
        assert_array_almost_equal(result.get_mean(), mean)
        assert_array_almost_equal(result.get_variance(), var)

    def test_data_stats_aggregation(self):
        pre_processor = NumericalPreProcessor()
        for _ in range(200):
            data_1 = np.random.random_sample((np.random.randint(1, 20), 10))*10.0
            data_2 = np.random.random_sample((np.random.randint(1, 20), 10))*10.0
            data_set_1 = NumericalDataSet(data_1, np.random.random_sample((data_1.shape[0],1)))
            data_set_2 = NumericalDataSet(data_2, np.random.random_sample((data_2.shape[0],1)))
            
            stats_1 = pre_processor.calculate(data_set_1)
            stats_2 = pre_processor.calculate(data_set_2)
            
            result = NumericalStats(encoded=pre_processor.aggregate(1, [stats_1, stats_2])['data'])
            
            data = np.vstack((data_set_1.inputs, data_set_2.inputs))
            size = data.shape[0]
            _min = np.min(data, axis=0)
            _max = np.max(data, axis=0)
            mean = np.mean(data, axis=0)
            var = np.var(data, axis=0)
            
            assert_equal(result.get_size(), size)
            assert_array_almost_equal(result.get_min(), _min)
            assert_array_almost_equal(result.get_max(), _max)
            assert_array_almost_equal(result.get_mean(), mean)
            assert_array_almost_equal(result.get_variance(), var)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()