'''
Created on Mar 4, 2015

@author: xapharius
'''
from validator.abstract_validator import AbstractValidator
import sklearn.metrics
import numpy as np

class ClassificationValidator(AbstractValidator):
    '''
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
        print len(predictions), predictions.sum()

        metrics["Accuracy"] = sklearn.metrics.accuracy_score(targets, predictions)
        metrics["Precision"] = sklearn.metrics.precision_score(targets, predictions, average="weighted")
        metrics["Recall"] = sklearn.metrics.recall_score(targets, predictions, average="weighted")
        metrics["F1"] = sklearn.metrics.f1_score(targets, predictions, average="weighted")
        

        return metrics

    def aggregate(self, validation_results):
        pass