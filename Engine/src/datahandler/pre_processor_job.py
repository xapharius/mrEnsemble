'''
Created on Mar 12, 2014

'''
import mrjob
from mrjob.job import MRJob

import reader
import pickle
import sys


class PreProcessorJob(MRJob):
    '''
    M/R job for pre-processing data. I.e. determine max, min, average, ...
    '''
    
    # defaults
    INPUT_PROTOCOL = reader.NLineCSVInputProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.NLineFileInputFormat'
    JOBCONF = { 'hadoopml.fileinput.linespermap': 200 }


    def init(self):
        
        data_handler = self._load_data_handler('data_handler.pkl')
        
        self.pre_processor = data_handler.get_pre_processor()
        self.data_processor = data_handler.get_data_processor()
        self.data_conf = data_handler.get_configuration()
        
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
        data_set = self.data_processor.get_data_set()
        yield 'stats', self.pre_processor.calculate(data_set)

    def reducer(self, key, values):
        vals = list(values)
        sys.stderr.write('reducer received: ' + str(vals) + '\n')
        yield key, self.pre_processor.aggregate(key, vals)

    def steps(self):
        return [
            self.mr( mapper_init  = self.init,
                     mapper       = self.mapper,
                     reducer_init = self.init,
                     reducer      = self.reducer )]

    def _load_data_handler(self, file_name):
        pkl_file = open(file_name, 'rb')
        data_handler = pickle.load(pkl_file)
        pkl_file.close()
        return data_handler

if __name__ == '__main__':
    PreProcessorJob.run()
