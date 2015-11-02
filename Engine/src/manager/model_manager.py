'''
Created on Mar 18, 2015

@author: xapharius
'''
import numpy as np
import time

class ModelManager(object):
    '''
    Abstracts the interaction between the feature selector and the model.
    Also keeps information about the "background" of a model - statistics about the training set.
    '''


    def __init__(self, model, feature_engineer):
        '''
        Constructor
        '''
        self.model = model
        self.feature_engineer = feature_engineer
        self.training_data_statistics = None
        self.training_performance = None

    def _calc_data_statistics(self, dataset):
        '''
        Get statistics about the data (training) in order to understand the models capabilities
        '''
        dct = {}
        dct["nr_obs"] = dataset.nrObservations
        return dct

    def _calc_performance(self, dataset):
        '''
        Calulates performance on dataset
        '''
        #TODO: some smart metrics here eg. partitioning hyperspace and get score for each cluster/bin
        return self.model.score(dataset.inputs, dataset.targets)

    def train(self, raw_inputs, targets=None, **kwargs):
        '''
        Fit model to data as well as gather training data and performance statistics 
        Must be inputs and targets (eg. for images)
        @param: raw_inputs: np.array, obs on columns
        @param: targets: np.arra, obs on columns
        '''
        start = time.time()
        dataset = self.feature_engineer.get_dataset(raw_inputs, targets)
        self.training_data_statistics = self._calc_data_statistics(dataset)
        self.model.fit(dataset.inputs, dataset.targets, **kwargs)
        print "Finished training in: {:.2f}s".format(time.time()-start)
        #self.training_performance = self._calc_performance(dataset)

    def predict(self, raw_inputs):
        dataset = self.feature_engineer.get_dataset(raw_inputs)
        prediction = self.model.predict(dataset.inputs)
        if len(prediction.shape) == 1:
            prediction = prediction.reshape(len(prediction), 1)
        return prediction


