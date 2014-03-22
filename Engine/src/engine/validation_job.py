'''
Created on Mar 19, 2014

@author: Simon
'''
import mrjob
from mrjob.job import MRJob

import protocol
import sys
from utils import serialization

class ValidationJob(MRJob):
    '''
    M/R job for validating a trained model.
    '''
    
    # defaults
    INPUT_PROTOCOL = protocol.NLineCSVInputProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.NLineFileInputFormat'
    JOBCONF = { 'hadoopml.fileinput.linespermap': 200 }


    def init(self):
        
        validation_objects = serialization.load_object('validation.pkl')
        
        self.data_handler = validation_objects['data_handler']
        self.pre_processor = self.data_handler.get_pre_processor()
        self.data_processor = self.data_handler.get_data_processor()
        self.data_conf = self.data_handler.get_configuration()
        
        self.validator = validation_objects['validator']
        self.alg = validation_objects['alg']
        
        # set configuration
        if self.data_conf:
            if self.data_conf.get_input_protocol():
                self.INPUT_PROTOCOL = self.data_conf.get_input_protocol()
            if self.data_conf.get_internal_protocol():
                self.INTERNAL_PROTOCOL = self.data_conf.get_internal_protocol()
            if self.data_conf.get_output_protocol():
                self.OUTPUT_PROTOCOL = self.data_conf.get_output_protocol()
            if self.data_conf.get_job_conf():
                self.JOBCONF = self.data_conf.get_job_conf()

    def mapper(self, key, values):
        self.data_processor.set_data(values)
        self.data_processor.normalize_data(self.data_handler.get_statistics())
        data_set = self.data_processor.get_data_set()
        yield 'validation', self.validator.validate(self.alg, data_set)

    def reducer(self, key, values):
        vals = list(values)
        sys.stderr.write('reducer received: ' + str(vals) + '\n')
        yield key, self.validator.aggregate(vals)

    def steps(self):
        return [
            self.mr( mapper_init  = self.init,
                     mapper       = self.mapper,
                     reducer_init = self.init,
                     reducer      = self.reducer )]


if __name__ == '__main__':
    ValidationJob.run()
