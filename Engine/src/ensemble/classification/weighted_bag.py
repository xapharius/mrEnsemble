'''
Created on Jul 29, 2015

@author: xapharius
'''

import numpy as np
import operator

class WBag(object):
    '''
    Weighted Bag of Classifiers
    '''

    def __init__(self, list_of_managers, list_of_weights=None):
        '''
        Constructor
        @param list_of_weights: None if equal weighted
        '''
        self.managers = list_of_managers
        if list_of_weights is None:
            self.weights = [1] * len(list_of_managers)
        else:
            self.weights = list_of_weights

    def set_weights(self, weights):
        assert len(weights) == len(self.managers)
        assert sum(weights) >= 0.99 and sum(weights) <= 1.0 
        self.weights = weights

    def _all_models_prediction(self, raw_inputs):
        """
        @return: list containing each model's prediction
        """
        predictions = [model.predict(raw_inputs) for model in self.managers]
        return predictions

    def _weighted_majority_voting(self, list_predictions):
        '''
        For one example :(
        '''
        target_len = len(list_predictions[0])
        if target_len == 1:
            predictions = list(np.ravel(list_predictions))
        else:
            # if binary sparse representation, use argmax
            predictions = [prediction.argmax() for prediction in list_predictions]
        dct_ix = {}
        for c in set(predictions):
            # manager belonging to each predicted class
            indices = [ix for ix, val in enumerate(predictions) if val == c]
            dct_ix[c] = indices
        dct_w = {}
        for key in dct_ix.keys():
            # sum of weights for each class
            dct_w[key] = sum(np.array(self.weights)[dct_ix[key]])
        # class with highest weight
        max_c = int(max(dct_w.iteritems(), key=operator.itemgetter(1))[0])
        if target_len != 1:
            # transform back to representation
            vec = [0] * target_len
            vec[max_c] = 1
            max_c = vec
        return max_c

    def predict(self, raw_inputs):
        # prediction for each model
        predictions_all = self._all_models_prediction(raw_inputs)
        # voting for each observation
        predictions_voted = []
        for i in range(len(raw_inputs)):
            predictions_all_obsi = [predictions[i] for predictions in predictions_all]
            predictions_voted.append(self._weighted_majority_voting(predictions_all_obsi))
        predictions_voted = np.vstack(np.array(predictions_voted)) # works for single labels aswell as arrays
        """
        predictions_voted = [self._weighted_majority_voting(list(predictions_all[i]))
            for i in range(len(predictions_all))]
        predictions_voted = np.array(predictions_voted).reshape(len(predictions_voted),1)
        """
        assert predictions_voted.shape == predictions_all[0].shape
        assert len(predictions_voted) == len(raw_inputs)
        return predictions_voted
