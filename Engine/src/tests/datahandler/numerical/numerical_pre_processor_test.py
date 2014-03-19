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
        
        tolerance = 0.001
        self.assertTrue(result[NumericalPreProcessor.NUM] == num, 'num doesn\'t match!\nex: ' + str(num) + '\nis: ' + str(result[NumericalPreProcessor.NUM]))
        self.assertTrue(all(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MIN] == _min), 'min doesn\'t match!\nex: ' + str(_min) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MIN]))
        self.assertTrue(all(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MAX] == _max), 'max doesn\'t match!\nex: ' + str(_max) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MAX]))
        self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MEAN], mean, tolerance), 'mean doesn\'t match!\nex: ' + str(mean) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MEAN]))
        self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.DATA][NumericalPreProcessor.VAR], var, tolerance), 'variance doesn\'t match!\nex: ' + str(var) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.VAR]))

    def testAggregation(self):
        pre_processor = NumericalPreProcessor()
        tolerance = 0.0001
        for _ in range(200):
            data_1 = np.random.random_sample((np.random.randint(1, 20), 10))*10.0
            data_2 = np.random.random_sample((np.random.randint(1, 20), 10))*10.0
            data_set_1 = NumericalDataSet(data_1, np.random.random_sample((data_1.shape[0],1)))
            data_set_2 = NumericalDataSet(data_2, np.random.random_sample((data_2.shape[0],1)))
            
            stats_1 = pre_processor.calculate(data_set_1)
            stats_2 = pre_processor.calculate(data_set_2)
            
            result = pre_processor.aggregate(1, [stats_1, stats_2])
            
            data = np.vstack((data_set_1.inputs, data_set_2.inputs))
            num = data.shape[0]
            _min = np.min(data, axis=0)
            _max = np.max(data, axis=0)
            mean = np.mean(data, axis=0)
            var = np.var(data, axis=0)
            
            self.assertTrue(result[NumericalPreProcessor.NUM] == num, 'num doesn\'t match!\nex: ' + str(num) + '\nis: ' + str(result[NumericalPreProcessor.NUM]))
            self.assertTrue(all(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MIN] == _min), 'min doesn\'t match!\nex: ' + str(_min) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MIN]))
            self.assertTrue(all(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MAX] == _max), 'max doesn\'t match!\nex: ' + str(_max) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MAX]))
            self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MEAN], mean, tolerance), 'mean doesn\'t match!\nex: ' + str(mean) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.MEAN]))
            self.assertTrue(equals_with_tolerance(result[NumericalPreProcessor.DATA][NumericalPreProcessor.VAR], var, tolerance), 'variance doesn\'t match!\nex: ' + str(var) + '\nis: ' + str(result[NumericalPreProcessor.DATA][NumericalPreProcessor.VAR]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()