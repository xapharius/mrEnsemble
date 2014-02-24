'''
Created on Feb 24, 2014

@author: Mihai
'''
import unittest
from algorithms.neuralnetwork.feedforward.PredictionNNFactory import PredictionNNFactory
from algorithms.neuralnetwork.feedforward.PredictionNN import PredictionNN
from numpy.ma.testutils import assert_equal
from numpy.ma.testutils import assert_array_equal
import numpy as np


class Test(unittest.TestCase):


    def test_init(self):
        layerSizes = [3,2,1]
        NNFactory = PredictionNNFactory(layerSizes)
        assert_equal(NNFactory.arrLayerSizes, layerSizes)
        
    def test_get_instance(self):
        '''test if return type of get_instance is correct
        '''
        layerSizes = [3,2,1]
        NNFactory = PredictionNNFactory(layerSizes)
        NN = NNFactory.get_instance()
        assert_equal(type(NN), PredictionNN)
    
    def test_aggregation(self):
        ''' test if aggregation, by averaging, works for a simple example
        '''
        layerSizes = [3,2,1]
        NNFactory = PredictionNNFactory(layerSizes)
        NNArr = []
        
        NN1 = NNFactory.get_instance()
        parameters = [];
        parameters.append(np.ones((4,2)))
        parameters.append(np.ones((3,1)))
        NN1.set_params(parameters)
        NNArr.append(NN1)
        
        NN2 = NNFactory.get_instance()
        parameters = [];
        parameters.append(3 * np.ones((4,2)))
        parameters.append(3 * np.ones((3,1)))
        NN2.set_params(parameters)
        NNArr.append(NN2)
        
        superNN = NNFactory.aggregate(NNArr)
        assert_array_equal(superNN.weightsArr[0], 2 * np.ones((4,2)))
        assert_array_equal(superNN.weightsArr[1], 2 * np.ones((3,1)))
    
    def test_serialize(self):
        pass
    
    def test_deserialize(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()