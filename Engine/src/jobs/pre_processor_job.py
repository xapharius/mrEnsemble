'''
Created on Mar 12, 2014

@author: Simon
'''

from engine.engine_job import EngineJob
from utils import logging


class PreProcessorJob(EngineJob):
    '''
    M/R job for pre-processing data. I.e. determine max, min, average, ...
    '''

    def mapper(self, key, values):
        logging.info("Pre processing " + str(len(values)) + " values")
        data_processor = self.get_data_processor()
        data_processor.set_data(values)
        data_set = data_processor.get_data_set()
        stats = self.get_pre_processor().calculate(data_set)
        logging.info("Calculated statistics: " + str(stats))
        yield 'stats', stats

    def reducer(self, key, values):
        vals = list(values)
        logging.info("Aggregating " + str(len(vals)) + " statistics")
        stats = self.get_pre_processor().aggregate(key, vals)
        logging.info("Aggregated statistics: " + str(stats))
        yield key, stats


if __name__ == '__main__':
    PreProcessorJob.run()
