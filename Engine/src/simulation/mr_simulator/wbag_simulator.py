'''
Created on Jul 30, 2015

@author: xapharius
'''

import numpy as np
import utils.multiproc as mp
from functools import partial
from ensemble.classification.weighted_bag import WBag

def train_map(model_factory, raw_data):
        '''
        train model instance for data slice
        '''
        inputs, targets = raw_data
        alg = model_factory.get_instance()
        alg.train(inputs, targets)
        return alg

class WBagSimulator(object):
    '''
    Idealized two-stepsimulation, ignoring the underlying technology. 
    This is used to test the algorithms in a controlled environment.
    For a simulation of the framework use the engine with _run_type = LOCAL
    DOESNT USE ENCODINGS..yet
    '''


    def __init__(self, data_sampler, factory, ensemble_cls=WBag):
        '''
        Constructor, Defining the environment
        @param data_sampler: already bound
        '''
        self.sampler = data_sampler
        self.model_factory = factory
        self.ensemble = ensemble_cls

    def simulate(self, nr_mappers):
        '''
        @param nr_mappers: number of independent samples
        @return the ensemble
        '''

        datasets = [self.sampler.sample() for _ in range(nr_mappers)] # data for each model

        # STEP 1
        map_func = mp.parallel_function(partial(train_map, self.model_factory))
        maps = map_func(datasets)
        #maps = [self.map_step1(data) for data in datasets]
        self.ensemble = self.reduce_step1(maps)

        # STEP 2
        maps = [self.map_step2(data) for data in datasets]
        ensemble = self.reduce_step2(maps)
        return ensemble

    def map_step1(self, raw_data):
        '''
        train model instance for data slice
        '''
        inputs, targets = raw_data
        alg = self.model_factory.get_instance()
        alg.train(inputs, targets)
        return alg

    def reduce_step1(self, maps):
        '''
        Create ensemble from list of models
        '''
        # 'values' is a generator, "convert" to list
        return self.ensemble(maps)

    def map_step2(self, raw_data_tuple):
        '''
        Get Extended prediction of ensemble for each data slice
        @return number of hits for each model
        '''
        inputs, targets = raw_data_tuple
        predictions = self.ensemble._all_models_prediction(inputs)
        hits_arr = []
        for prediction in predictions:
            hits = 0
            for ix, row in enumerate(prediction):
                hits += 1 if (row == targets[ix]).all() else 0
            hits_arr.append(hits)
        return np.array(hits_arr)

    def reduce_step2(self, maps):
        '''
        Compute weights from performance of models on each slice
        '''
        ensemble = self.ensemble
        performance = np.vstack(maps)
        assert performance.shape == (len(maps), len(maps))
        performance = performance.sum(axis=0)
        weights = performance / float(performance.sum())
        ensemble.set_weights(weights)
        return ensemble
