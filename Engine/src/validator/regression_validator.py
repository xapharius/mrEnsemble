'''
Created on Mar 3, 2015

@author: xapharius
'''

from abstract_validator import AbstractValidator
import sklearn.metrics

class RegressionValidator(AbstractValidator):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        pass

    def validate(self, alg, inputs, targets):
        '''
        @param inputs: raw inputs
        '''
        metrics = {}
        predictions = alg.predict(inputs)

        metrics["Explained Variance"] = sklearn.metrics.explained_variance_score(targets, predictions)
        metrics["R2 Score"] = sklearn.metrics.r2_score(targets, predictions)
        metrics["Mean Absolute Error"] = sklearn.metrics.mean_absolute_error(targets, predictions)
        metrics["Mean Squared Error"] = sklearn.metrics.mean_squared_error(targets, predictions)

        return metrics

    def aggregate(self, validation_results):
        pass