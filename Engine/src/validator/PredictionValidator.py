'''
Created on Mar 12, 2014

@author: xapharius
'''
import utils.numpyutils as nputils
import numpy as np
import operator
from validator.abstract_validator import AbstractValidator
from utils import numpyutils

class PredictionValidator(AbstractValidator):
    '''
    Class that establishes metrics for trained Prediction Model to measure it's performance.
    '''

    def validate(self, alg, data_set):
        predictions = alg.predict(data_set)
        mse = self.compute_mse(data_set.get_targets(), predictions)
#         r2 = self.compute_r2(data_set.get_targets(), predictions)
        # predictions are a list of 2D numpy arrays
        # convert to 2D matrix
        prediction_matrix = []
        for pred in predictions:
            prediction_matrix.extend(numpyutils.to_list(pred))
        return { 'mse': mse, 'num': data_set.get_nr_observations(), 'targets': data_set.get_targets().tolist(), 'pred': prediction_matrix }

    def aggregate(self, validation_results):
        result = { 'mse': 0, 'num': 0, 'targets': [], 'pred': [] }
        for r in validation_results:
            if result['num'] == 0:
                result['mse'] = r['mse']
                result['num'] = r['num']
                result['targets'] = r['targets']
                result['pred'] = r['pred']
            else:
                result['mse'] = (result['mse']*result['num'] + r['mse']*r['num'])/(result['num'] + r['num'])
                result['num'] += r['num']
                result['targets'].extend(r['targets'])
                result['pred'].extend(r['pred'])
        return result

    # TODO: prediction with multiple outputs?
    def compute_mse(self, targets, predictions):
        '''
        Computes the Mean Squared Error
        @param targets: list of np.arrays
        @param predictions: list of np.arrays
        @rtype: scalar
        '''
        return np.sum(nputils.expNPArrayList(map(operator.sub, targets, predictions), 2))/len(predictions)
    
    # TODO: prediction with multiple outputs?
    # TODO: SST != SSR + SSE
    def compute_r2(self, targets, predictions):
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
         

# TODO: Do we need this?
#     def validate_local(self, model, dataSet):
#         '''
#         Establishing performance metrics for given dataSet (eg testing set)
#         @param model: model inheriting AbstractAlgorithm
#         @param dataSet: dataset inheriting AbstractDataSet  
#         '''
#         predictions = model.predict(dataSet)
#         _, targets = zip(*dataSet.gen_observations())
#     
#         print("Mean Squared Error:", self.computeMSE(targets, predictions)[0,0])
#         print("R-Squared: ", self.computeR2(targets, predictions)[0,0])
#         
#     # TODO: validate_hdfs
#     def validate_hdfs(self, model, hdfs_file):
#         '''
#         Establishing performance metrics for file found on hdfs (eg training set)
#         @param model: model inheriting AbstractAlgorithm
#         @param dataSet: dataset inheriting AbstractDataSet  
#         '''
#         pass
