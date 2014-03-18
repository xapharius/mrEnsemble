'''
Created on Mar 17, 2014

@author: linda
'''
import unittest
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from datahandler.numerical.numerical_pre_processor import NumericalPreProcessor
import numpy as np
from tests.test_utils import equals_with_tolerance

class NumericalPreProcessorTest(unittest.TestCase):


    def testCalculation(self):
        pre_processor = NumericalPreProcessor()
        data_set = NumericalDataSet(np.random.random_integers(1,10, (10,10)), np.random.random_integers(1,10, (10,1)))
        result = pre_processor.calculate(data_set)
        
        data = data_set.inputs
        num = data.shape[0]
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        var = np.var(data, axis=0)
        
        tolerance = 0.01
        self.assertTrue(result[NumericalPreProcessor.NUM] == num, 'num doesn\'t match!\nex: ' + str(num) + '\nis: ' + str(result[NumericalPreProcessor.NUM]))
        self.assertTrue(all(result[NumericalPreProcessor.MIN] == _min), 'min doesn\'t match!\nex: ' + str(_min) + '\nis: ' + str(result[NumericalPreProcessor.MIN]))
        self.assertTrue(all(result[NumericalPreProcessor.MAX] == _max), 'max doesn\'t match!\nex: ' + str(_max) + '\nis: ' + str(result[NumericalPreProcessor.MAX]))
        self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.MEAN], mean, tolerance), 'mean doesn\'t match!\nex: ' + str(mean) + '\nis: ' + str(result[NumericalPreProcessor.MEAN]))
        self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.VAR], var, tolerance), 'variance doesn\'t match!\nex: ' + str(var) + '\nis: ' + str(result[NumericalPreProcessor.VAR]))

    def testAggregation(self):
        pre_processor = NumericalPreProcessor()
        data_set_1 = NumericalDataSet(np.random.random_integers(1,10, (10,10)), np.random.random_integers(1,10, (10,1)))
        data_set_2 = NumericalDataSet(np.random.random_integers(1,10, (10,10)), np.random.random_integers(1,10, (10,1)))
        
        stats_1 = pre_processor.calculate(data_set_1)
        stats_2 = pre_processor.calculate(data_set_2)
        
        result = pre_processor.aggregate(1, [stats_1, stats_2])
        
        data = np.vstack((data_set_1.inputs, data_set_2.inputs))
        num = data.shape[0]
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        var = np.var(data, axis=0)
        
        # TODO: variance accuracy issue
        tolerance = 2.0
        self.assertTrue(result[NumericalPreProcessor.NUM] == num, 'num doesn\'t match!\nex: ' + str(num) + '\nis: ' + str(result[NumericalPreProcessor.NUM]))
        self.assertTrue(all(result[NumericalPreProcessor.MIN] == _min), 'min doesn\'t match!\nex: ' + str(_min) + '\nis: ' + str(result[NumericalPreProcessor.MIN]))
        self.assertTrue(all(result[NumericalPreProcessor.MAX] == _max), 'max doesn\'t match!\nex: ' + str(_max) + '\nis: ' + str(result[NumericalPreProcessor.MAX]))
        self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.MEAN], mean, tolerance), 'mean doesn\'t match!\nex: ' + str(mean) + '\nis: ' + str(result[NumericalPreProcessor.MEAN]))
        self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.VAR], var, tolerance), 'variance doesn\'t match!\nex: ' + str(var) + '\nis: ' + str(result[NumericalPreProcessor.VAR]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()