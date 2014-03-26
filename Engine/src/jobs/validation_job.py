'''
Created on Mar 19, 2014

@author: Simon
'''

from engine.engine_job import EngineJob

class ValidationJob(EngineJob):
    '''
    M/R job for validating a trained model.
    '''

    def mapper(self, key, values):
        data_processor = self.get_data_processor()
        data_processor.set_data(values)
        data_processor.normalize_data(self.data_handler.get_statistics())
        data_set = data_processor.get_data_set()
        alg = self.get_trained_alg()
        validator = self.get_validator()
        yield 'validation', validator.validate(alg, data_set)

    def reducer(self, key, values):
        vals = list(values)
        yield key, self.get_validator().aggregate(vals)


if __name__ == '__main__':
    ValidationJob.run()
