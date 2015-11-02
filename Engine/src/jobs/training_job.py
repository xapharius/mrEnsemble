'''
Created on Mar 22, 2014

@author: linda
'''
from engine.engine_job import EngineJob
from utils import logging

class TrainingJob(EngineJob):
    '''
    M/R Job for the actual training of an algorithm instance.
    '''

    def mapper(self, key, values):
        # create new algorithm instance
        alg_factory = self.get_alg_factory()
        alg = alg_factory.get_instance()
        # create normalized data set
        data_processor = self.get_data_processor()
        data_processor.set_data(values)
        data_processor.normalize_data(self.get_statistics())
        data_set = data_processor.get_data_set()
        # train the model
        alg.train(data_set)
        # prepare algorithm for transport
        serialized = alg_factory.encode(alg)
        yield 'alg', serialized

    def reducer(self, key, values):
        alg_factory = self.get_alg_factory()
        # 'values' is a generator, "convert" to list
        values_list = list(values)
        alg = alg_factory.aggregate(alg_factory.decode(values_list))
        yield 'alg', alg_factory.encode(alg)


if __name__ == '__main__':
    TrainingJob.run()