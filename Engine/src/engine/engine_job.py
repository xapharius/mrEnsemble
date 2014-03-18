import mrjob
from mrjob.job import MRJob

import protocol
import sys
import pickle


class EngineJob(MRJob):
    '''
    M/R Job for the actual training of an algorithm instance.
    '''
    
    INPUT_PROTOCOL = protocol.NLineCSVInputProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.NLineFileInputFormat'
    JOBCONF = { 'hadoopml.fileinput.linespermap': 200 }


    def init(self):
        # load data handler and algorithm factory
        self.factory = self._load_object('alg_factory.pkl')
        self.data_handler = self._load_object('data_handler.pkl')
        self.data_conf = self.data_handler.get_configuration()
        
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

    def mapper(self, key, value):
        # create new algorithm instance
        alg = self.factory.get_instance()
        # create normalized data set
        data_processor = self.data_handler.get_data_processor()
        data_processor.set_data(value)
        data_processor.normalize_data(self.data_handler.get_statistics())
        data_set = data_processor.get_data_set()
        # train the model
        alg.train(data_set)
        sys.stderr.write('alg params: ' + str(alg.params) + '\n')
        # prepare algorithm for transport
        serialized = self.factory.encode(alg)
        sys.stderr.write('serialized: ' + str(serialized) + '\n')
        yield 0, serialized

    def reducer(self, key, values):
        # 'values' is a generator, "convert" to list
        values_list = list(values)
        sys.stderr.write("reducer: \n  key: " + str(key) + "\n  value: " + str(values_list) + "\n")
        alg = self.factory.aggregate(self.factory.decode(values_list))
        yield 0, self.factory.encode(alg)

    def steps(self):
        return [
            self.mr( mapper_init  = self.init,
                     mapper       = self.mapper,
                     reducer_init = self.init,
                     reducer      = self.reducer )]

    def _load_object(self, file_name):
        pkl_file = open(file_name, 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()
        return obj

if __name__ == '__main__':
    EngineJob.run()
