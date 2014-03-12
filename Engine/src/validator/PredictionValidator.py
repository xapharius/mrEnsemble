'''
Created on Mar 12, 2014

@author: xapharius
'''
import utils.numpyutils as nputils
import numpy as np
import operator

class PredictionValidator(object):
    '''
    Class that establishes metrics for trained Prediction Model to measure it's performance.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    # TODO: prediction with multiple outputs?
    def computeMSE(self, targets, predictions):
        '''
        Computes the Mean Squared Error
        @param targets: list of np.arrays
        @param predictions: list of np.arrays
        @rtype: scalar
        '''
        return sum(nputils.expNPArrayList(map(operator.sub, targets, predictions), 2))/len(predictions)
    
    # TODO: prediction with multiple outputs?
    # TODO: SST != SSR + SSE
    def computeR2(self, targets, predictions):
        '''
        Computes R-squared
        @param targets: list of np.arrays
        @param predictions: list of np.arrays
        @rtype: scalar
        '''
        residuals = map(operator.sub, targets, predictions)
        # sum of squared residuals
        sse = sum(nputils.expNPArrayList(residuals, 2))
        # variance of targets
        sst = np.var(targets)*len(targets)
        return 1 - sse/sst
         
    
    def validate_local(self, model, dataSet):
        '''
        Establishing performance metrics for given dataSet (eg testing set)
        @param model: model inheriting AbstractAlgorithm
        @param dataSet: dataset inheriting AbstractDataSet  
        '''
        predictions = model.predict(dataSet)
        _, targets = zip(*dataSet.gen_observations())
    
        print("Mean Squared Error:", self.computeMSE(targets, predictions)[0,0])
        print("R-Squared: ", self.computeR2(targets, predictions)[0,0])
        
    # TODO: validate_hdfs
    def validate_hdfs(self, model, hdfs_file):
        '''
        Establishing performance metrics for file found on hdfs (eg training set)
        @param model: model inheriting AbstractAlgorithm
        @param dataSet: dataset inheriting AbstractDataSet  
        '''
        pass
