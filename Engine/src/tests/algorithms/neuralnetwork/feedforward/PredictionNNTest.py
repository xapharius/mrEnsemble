'''
Created on Feb 24, 2014

@author: xapharius
'''
import unittest
from algorithms.neuralnetwork.feedforward.PredictionNN import PredictionNN
from numpy.ma.testutils import assert_equal
import numpy as np
import sys
import utils.numpyutils as nputils
from datahandler.numerical.NumericalDataSet import NumericalDataSet


class Test(unittest.TestCase):

    def test_constructor_medium_nn(self):
        '''
        Testing if members initialized correctly.
        3 - layer NN
        '''
        layerSizes = [3,2,1]
        NN = PredictionNN(layerSizes)
        # n layers -> n - 1 weight matrices
        assert_equal(len(NN.weightsArr), len(layerSizes) - 1, "number of layers not equal")
        assert_equal(NN.arrLayerSizes, layerSizes, "layerSize member not right size")
        for i in range(len(layerSizes) - 1):
            assert_equal(NN.weightsArr[i].shape, (layerSizes[i] + 1, layerSizes[i+1]), "weight Matrix size not equal")
    
    def test_constructor_small_nn(self):
        '''
        Testing if members initialized correctly.
        3 - layer NN
        '''  
        layerSizes = [2]
        NN = PredictionNN(layerSizes)
        # n layers -> n - 1 weight matrices
        assert_equal(len(NN.weightsArr), len(layerSizes) - 1, "number of layers not equal")
        assert_equal(NN.arrLayerSizes, layerSizes, "layerSize member not right size")
        for i in range(len(layerSizes) - 1):
            assert_equal(NN.weightsArr[i].shape, (layerSizes[i] + 1, layerSizes[i+1]), "weight Matrix size not equal")
            
    def test_constructor_big_nn(self):
        '''
        Testing if members initialized correctly.
        9 - layer NN
        ''' 
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
        except Exception, e: 
            expected_errmsg = "overwriting parameters have not the same shape as the model"
            assert(e.message.startswith(expected_errmsg))
        else:
            self.fail("no exception thrown")
    
    def test_feedforward(self):
        '''
        test feed forward for sample input. activations should coincide with calculated values
        shape (1,x)
        '''
        layerSizes = [2,2,1]
        nn = PredictionNN(layerSizes)
        
        parameters = [];
        parameters.append(np.ones((3,2)))
        parameters.append(np.ones((3,1)))
        nn.set_params(parameters)
        
        inputVec = np.array([[2,2]])
        activations = nn.feedforward(inputVec)
        assert_equal(activations[0], np.array([[2,2]]), "input activations wrong")
        assert_equal(activations[1], np.array([[0.9933071490757153, 0.9933071490757153]]), "hidden activations wrong")
        assert_equal(activations[2], np.array([[2.9866142981514305]]), "output activations wrong")
    
    def test_backprop(self):
        layerSizes = [1,1,2,1]
        nn = PredictionNN(layerSizes)
        parameters = [];
        parameters.append(np.ones((2,1)))
        parameters.append(np.ones((2,2)))
        parameters.append(np.ones((3,1)))
        nn.set_params(parameters)
        
        inputVec = np.array([[1]])
        activations = nn.feedforward(inputVec)
        nn.backpropagation(activations, np.array([0.5]))
        
        # value computed through backpropagation by hand
        assert_equal(round(nn.weightsArr[0][0,0], 7), 0.9973057)
    
    def test_train(self):
        '''
        Testing only execution of train function
        '''
        layerSizes = [2,2,1]
        nn = PredictionNN(layerSizes)
        
        # preparing input NumericalDataSet
        inputSet = np.array([[2,2]])
        inputVec = np.array([[2,2]])
        targetSet = np.array([[1]])
        targetVec = np.array([[1]])
        nrObs = 10
        for _ in range(nrObs-1):
            inputSet = np.vstack((inputSet,inputVec))
            targetSet = np.vstack((targetSet,targetVec))
        dataSet = NumericalDataSet(inputSet, targetSet)
        nn.train(dataSet)
        
        
    
    def test_predict(self):
        '''
        Test prediction for dataset. Returns a list of np.arrays.
        '''
        # defining model
        layerSizes = [2,2,1]
        nn = PredictionNN(layerSizes)
        
        # setting up nn
        parameters = [];
        parameters.append(np.ones((3,2)))
        parameters.append(np.ones((3,1)))
        nn.set_params(parameters)
        
        # preparing input NumericalDataSet
        inputSet = np.array([[2,2]])
        inputVec = np.array([[2,2]])
        nrObs = 10
        for _ in range(nrObs-1):
            inputSet = np.vstack((inputSet,inputVec))
        dataSet = NumericalDataSet(inputSet, None)
        
        # run function
        predictions = nn.predict(dataSet)
        
        # check nr of observations
        self.assertEqual(len(predictions), nrObs, "number of observations mismatch")
        for prediction in predictions:
            assert_equal(prediction, np.array([[2.9866142981514305]]), "wrong output")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()