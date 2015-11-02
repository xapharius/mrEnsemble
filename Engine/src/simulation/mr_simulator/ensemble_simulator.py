'''
Created on Mar 23, 2015

@author: xapharius
'''


class EnsembleSimulator(object):
    '''
    Idealized simulation, ignoring the underlying technology. 
    This is used to test the algorithms in a controlled environment.
    For a simulation of the framework use the engine with _run_type = LOCAL
    DOESNT USE ENCODINGS..yet
    '''


    def __init__(self, data_sampler, factory, ensemble_cls):
        '''
        Constructor, Defining the environment
        '''
        self.sampler = data_sampler
        self.model_factory = factory
        self.ensemble = ensemble_cls

    def simulate(self, nr_mappers):
        '''
        @param nr_mappers: number of independent samples
        @return the ensemble
        '''

        # map
        maps = [] #array of models
        for _ in range(nr_mappers):
            data = self.sampler.sample()
            maps.append(self.map_step(data))

        # reduce
        ensemble = self.reduce_step(maps) #is still encoded

        return ensemble

    def map_step(self, raw_data):
        inputs, targets = raw_data
        alg = self.model_factory.get_instance()
        alg.train(inputs, targets)
        return alg

    def reduce_step(self, model_list):
        # 'values' is a generator, "convert" to list
        return self.ensemble(model_list)
