'''
Created on Feb 12, 2015

@author: xapharius
'''
from timeit import itertools

class Simulator(object):
    '''
    Idealized simulator, ignoring the underlying technology. 
    This is used to test the algorithms in a controlled environment.
    For a simulation of the framework use the engine with _run_type = LOCAL
    '''


    def __init__(self, data_sampler, data_handler, algorithm_factory):
        '''
        Constructor, Defining the environment
        '''
        self.sampler = data_sampler
        self.data_handler = data_handler
        self.alg_factory = algorithm_factory

    def simulate(self, nr_mappers):
        '''
        @param nr_mappers: number of independent samples
        @return the ensemble
        '''

        # map
        maps = []
        for _ in range(nr_mappers):
            data = self.sampler.sample()
            maps.append(self.map_step(data))

        # reduce
        encoded_alg = self.reduce_step(maps) #is still encoded
        result_alg = self.alg_factory.decode([encoded_alg])[0]

        return result_alg


    def map_step(self, values):

        data_processor = self.data_handler.get_data_processor()
        data_processor.set_data(values)
        data_set = data_processor.get_data_set()

        alg = self.alg_factory.get_instance()
        alg.train(data_set)
        # prepare algorithm for transport
        serialized = self.alg_factory.encode(alg)
        return serialized

    def reduce_step(self, values):
        # 'values' is a generator, "convert" to list
        values_list = list(values)
        alg = self.alg_factory.aggregate(self.alg_factory.decode(values_list))
        return self.alg_factory.encode(alg)






