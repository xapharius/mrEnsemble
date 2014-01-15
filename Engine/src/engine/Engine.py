import mrjob
from mrjob.job import MRJob

from algorithms.linearRegression.LinearRegressionFactory import \
    LinearRegressionFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
import reader


class Engine(MRJob):
    '''
    classdocs
    '''
    
    INPUT_PROTOCOL = reader.NLineCSVInputProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.NLineFileInputFormat'
    JOBCONF = { 'hadoopml.fileinput.linespermap': 2 }
 
    def init(self, factory, data_handler):
        self.factory = factory
        self.data_handler = data_handler
         
        # set configuration
        if self.data_handler.get_configuration():
            self.conf = data_handler.get_configuration()
            if self.conf.get_input_protocol():
                self.INPUT_PROTOCOL = self.conf.get_input_protocol()
            if self.conf.get_internal_protocol():
                self.INTERNAL_PROTOCOL = self.conf.get_internal_protocol()
            if self.conf.get_output_protocol():
                self.OUTPUT_PROTOCOL = self.conf.get_output_protocol()
            if self.conf.get_job_conf():
                self.JOBCONF = self.conf.get_job_conf() 
    
    def mapper(self, key, value):
        self.init(LinearRegressionFactory(11), NumericalDataHandler(11, 1))
        alg = self.factory.get_instance()
        data_set = self.data_handler.get_DataProcessor(value).get_data()
        alg.train(data_set)
        # TODO: serialize algorithm (parameters of course)
        yield key, self.factory.serialize(alg)
    
    def reducer(self, key, value):
        # TODO: value should be np.array of type LinearRegression
        # TODO: serialize algorithm (parameters of course)
        alg = self.factory.aggregate(self.factory.deserialize(value))
        yield key, self.factory.serialize(alg) 

    def steps(self):
        return [
            self.mr(mapper=self.mapper,
                    reducer=self.reducer)]

if __name__ == '__main__':
    
    nrParams = 11
    nrLabelDim = 1
    
    factory = LinearRegressionFactory(nrParams)
    data_handler = NumericalDataHandler(nrParams, nrLabelDim) 
    engine = Engine()
    engine.init(factory, data_handler)
    
    engine.run()