'''
Created on Mar 12, 2014

@author: Simon
'''

from engine.engine_job import EngineJob


class PreProcessorJob(EngineJob):
    '''
    M/R job for pre-processing data. I.e. determine max, min, average, ...
    '''

    def mapper(self, key, values):
        data_processor = self.get_data_processor()
        data_processor.set_data(values)
        data_set = data_processor.get_data_set()
        yield 'stats', self.get_pre_processor().calculate(data_set)

    def reducer(self, key, values):
        vals = list(values)
        yield key, self.get_pre_processor().aggregate(key, vals)


if __name__ == '__main__':
    PreProcessorJob.run()
