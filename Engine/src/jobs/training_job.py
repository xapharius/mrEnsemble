'''
Created on Mar 22, 2014

@author: Simon
'''

import numpy as np

class TrainingJob():
    '''
    M/R Job for the actual training of an algorithm instance.
    '''
    def __init__(self, engine):
        self.engine = engine

    def mapper(self, key, values):
        data = np.array(values[0])
        stats = values[1]
        # create new algorithm instance
        alg_factory = self.engine.get_alg_factory()
        alg = alg_factory.get_instance()
        # create normalized data set
        data_processor = self.engine.get_data_processor()
        data_processor.set_data(data)
        data_processor.normalize_data(stats)
        data_set = data_processor.get_data_set()
        # train the model
        alg.train(data_set)
        # prepare algorithm for transport
        serialized = alg_factory.encode(alg)
        yield 'alg', (serialized, stats)

    def reducer(self, key, values):
        alg_factory = self.engine.get_alg_factory()
        # 'values' is a generator, "convert" to list
        values_list = list(values)
        algs = [ t[0] for t in values_list ]
        stats = [ t[1] for t in values_list ]
        alg = alg_factory.aggregate(alg_factory.decode(algs))
        yield 'alg', (alg_factory.encode(alg), stats[0])
