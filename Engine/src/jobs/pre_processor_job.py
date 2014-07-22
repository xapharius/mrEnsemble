'''
Created on Mar 12, 2014

@author: Simon
'''

from utils import logging


class PreProcessorJob():
    '''
    M/R job for pre-processing data. I.e. determine max, min, average, ...
    '''
    
    def __init__(self, engine):
        self.engine = engine

    def mapper(self, key, values):
        logging.info("Pre processing " + str(len(values)) + " values")
        data_processor = self.engine.get_data_processor()
        data_processor.set_data(values)
        data_set = data_processor.get_data_set()
        stats = self.engine.get_pre_processor().calculate(data_set)
        logging.info("Calculated statistics: " + str(stats))
        yield 'stats', ((key, values.tolist()), stats)

    def reducer(self, key, values):
        vals = list(values)
        data = [ t[0] for t in vals ]
        stats = [ t[1] for t in vals ]
        logging.info("Aggregating " + str(len(stats)) + " statistics")
        aggregated_stats = self.engine.get_pre_processor().aggregate(key, stats)
        logging.info("Aggregated statistics: " + str(aggregated_stats))
        # yield all data like the mapper received it and include the aggregated
        # statistics, yielding the data with the original key will allow the 
        # distribution of the data to multiple nodes again
        for d in data:
            yield d[0], (d[1], aggregated_stats)
