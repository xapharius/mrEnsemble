from mrjob.job import MRJob
from engine_job import EngineJob
import sys
from pre_processor_job import PreProcessorJob
from validation_job import ValidationJob
from utils import serialization


class Engine(MRJob):

    DATA_HANDLER_FILE_NAME = 'data_handler.pkl'
    ALG_FACTORY_FILE_NAME = 'alg_factory.pkl'


    def __init__(self, alg_factory, data_file, data_handler=None, data_handler_file=None):
        '''
        @param alg_factory: algorithm factory implementing AbstractAlgorithmFactory
        @param data_handler: data handler implementing AbstractDataHandler
        @param data_file: path (string) to data (either local or HDFS)
        @param data_handler_file: Optional, if set the data handler is loaded
        from the file with the given file name omitting pre-processing.
        '''
        self.alg_factory = alg_factory
        if data_handler_file is not None:
            self.data_handler = serialization.load_object(data_handler_file)
            self.skip_pre_processing = True
        elif data_handler is not None:
            self.data_handler = data_handler
            self.skip_pre_processing = False
        else:
            raise Exception("Either a data handler instance or a data handler file name have to be given!")

        self.data_file = data_file


    def start(self):

        # ---------- Setup ----------------------------------------------------

        # serialize data handler and factory
        print('Serializing data handler and algorithm factory...')
        serialization.save_object(self.DATA_HANDLER_FILE_NAME, self.data_handler)
        serialization.save_object(self.ALG_FACTORY_FILE_NAME, self.alg_factory)
        print('...Done.\n')


        # -------- Pre-Processing ---------------------------------------------

        if not self.skip_pre_processing:
            pre_processor_job = PreProcessorJob(args=[
                      self.data_file,
                      '-r',               'hadoop',
                      '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                      '--file',           self.DATA_HANDLER_FILE_NAME,
                      '--hadoop-arg',     '-libjars',
                      '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                      '--python-archive', 'target/hadoop_ml.tar.gz',
                      '--strict-protocols'])
    
            # create output protocol instance for output parsing
            output_protocol = pre_processor_job.OUTPUT_PROTOCOL()
    
            with pre_processor_job.make_runner() as runner:
                print('Running pre-processing job...')
                runner.run()
                print('...Done.\n')
                # get pre-processing results and update data handler
                # (!) assuming there is only a single result
                (_, stats) = output_protocol.read(runner.stream_output().next())
                self.data_handler.set_statistics(stats)
                # overwrite old data handler file
                serialization.save_object(self.DATA_HANDLER_FILE_NAME, self.data_handler)
        else:
            print("Skipping pre-processing\n")


        # ----------- Training ------------------------------------------------

        engine_job = EngineJob(args=[
                  self.data_file,
                  '-r',               'hadoop',
                  '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--file',           self.DATA_HANDLER_FILE_NAME,
                  '--file',           self.ALG_FACTORY_FILE_NAME,
                  '--hadoop-arg',     '-libjars',
                  '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--python-archive', 'target/hadoop_ml.tar.gz',
                  '--strict-protocols'])
        output_protocol = engine_job.OUTPUT_PROTOCOL()
        with engine_job.make_runner() as runner:
            print('Running training job...') 
            runner.run()
            print('...Done.\n')
            # (!) assuming there is only a single result
            (_, trained_alg) = output_protocol.read(runner.stream_output().next())
        
        # TODO: cleanup pkl files
        
        return self.alg_factory.decode([trained_alg])[0]


    def validate(self, alg, validator):
        '''
        @param alg: Trained algorithm
        @param validator: Validator which should be used for validating a trained model.
        '''
        validation_objects = { 'data_handler': self.data_handler, 'alg': alg, 'validator': validator }
        serialization.save_object('validation.pkl', validation_objects)
        
        validation_job = ValidationJob(args=[
                  self.data_file,
                  '-r',               'hadoop',
                  '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--file',           'validation.pkl',
                  '--hadoop-arg',     '-libjars',
                  '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--python-archive', 'target/hadoop_ml.tar.gz',
                  '--strict-protocols'])
        output_protocol = validation_job.OUTPUT_PROTOCOL()
        print('Running validation job...') 
        with validation_job.make_runner() as runner:
            runner.run()
            print('...Done.\n')
            # (!) assuming there is only a single result
            (_, stats) = output_protocol.read(runner.stream_output().next())
        return stats


    def save_data_handler(self, file_name):
        serialization.save_object(file_name, self.data_handler)


def class_from_name(name):
    return getattr(sys.modules[__name__], name)
