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

    def validate(self, alg, dataset):
        metrics = {}
        predictions = alg.predict(dataset)
        targets = dataset.targets

        metrics["Accuracy"] = sklearn.metrics.accuracy_score(targets, predictions)
        metrics["F1"] = sklearn.metrics.f1_score(targets, predictions)
        metrics["Precision"] = sklearn.metrics.precision_score(targets, predictions)
        metrics["Recall"] = sklearn.metrics.recall_score(targets, predictions)
        if len(np.unique(targets)) == 2:
            # works only for binary targets
            metrics["Average Precision"] = sklearn.metrics.average_precision_score(targets, predictions)
            metrics["ROC_AUC"] = sklearn.metrics.roc_auc_score(targets, predictions)

        return metrics

    def aggregate(self, validation_results):
        pass