'''
Created on Feb 24, 2014

@author: Mihai
'''
import unittest
from algorithms.neuralnetwork.feedforward.PredictionNN import PredictionNN
from numpy.ma.testutils import assert_equal
import numpy as np
import sys


class PredictionNNTest(unittest.TestCase):

    # integrity test for class members
    def test_init1(self):
        layerSizes = [3,2,1]
        NN = PredictionNN(layerSizes)
        # n layers -> n - 1 weight matrices
        assert_equal(len(NN.weightsArr), len(layerSizes) - 1, "number of layers not equal")
        assert_equal(NN.arrLayerSizes, layerSizes, "layerSize member not right size")
        for i in range(len(layerSizes) - 1):
            assert_equal(NN.weightsArr[i].shape, (layerSizes[i] + 1, layerSizes[i+1]), "weight Matrix size not equal")
    
    def test_init2(self):   
        layerSizes = [2]
        NN = PredictionNN(layerSizes)
        # n layers -> n - 1 weight matrices
        assert_equal(len(NN.weightsArr), len(layerSizes) - 1, "number of layers not equal")
        assert_equal(NN.arrLayerSizes, layerSizes, "layerSize member not right size")
        for i in range(len(layerSizes) - 1):
            assert_equal(NN.weightsArr[i].shape, (layerSizes[i] + 1, layerSizes[i+1]), "weight Matrix size not equal")
            
    def test_init3(self):
        layerSizes = [1,2,3,4,5,6,7,8,9]
        NN = PredictionNN(layerSizes)
        # n layers -> n - 1 weight matrices
        assert_equal(len(NN.weightsArr), len(layerSizes) - 1, "number of layers not equal")
        assert_equal(NN.arrLayerSizes, layerSizes, "layerSize member not right size")
        for i in range(len(layerSizes) - 1):
            assert_equal(NN.weightsArr[i].shape, (layerSizes[i] + 1, layerSizes[i+1]), "weight Matrix size not equal")

    def test_set_params_positive(self):
        layerSizes = [3,2,1]
        NN = PredictionNN(layerSizes)
        
        parameters = [];
        parameters.append(np.ones((4,2)))
        parameters.append(np.ones((3,1)))
        NN.set_params(parameters)
        
    def test_set_params_negative(self):
        layerSizes = [3,2,1]
        NN = PredictionNN(layerSizes)
        
        parameters = [];
        parameters.append(np.ones((4,2)))
        parameters.append(np.ones((2,1)))
        try:
            NN.set_params(parameters)
            self.fail("no exception thrown")
        except: 
            expected_errmsg = "overwriting parameters have not the same shape as the model"
            errmsg = str(sys.exc_info()[1])
            assert(errmsg.startswith(expected_errmsg))
        
    
    def test_feedforward(self):
        pass
    
    def test_backprop(self):
        pass
    
    def test_train(self):
        pass
    
    def test_predict(self):
        pass
    
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()