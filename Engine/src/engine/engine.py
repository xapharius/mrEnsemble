from mrjob.job import MRJob
import sys
from jobs.pre_processor_job import PreProcessorJob
from jobs.validation_job import ValidationJob
from utils import serialization
from jobs.training_job import TrainingJob
import constants.internal as const


class Engine(MRJob):

    def __init__(self, alg_factory, data_file, data_handler=None, data_handler_file=None, run_type='hadoop'):
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
        self.run_type = run_type

    def start(self):

        # ---------- Setup ----------------------------------------------------

        # serialize data handler and factory
        print('Serializing data handler and algorithm factory...')
        self.conf = { const.DATA_HANDLER: self.data_handler, const.ALG_FACTORY: self.alg_factory }
        serialization.save_object(const.CONF_FILE_NAME, self.conf)
        print('...Done.\n')


        # -------- Pre-Processing ---------------------------------------------

        if self.skip_pre_processing:
            print("Skipping pre-processing\n")
        else:
            pre_processor_job = PreProcessorJob(args=[
                      self.data_file,
                      '-r',               self.run_type,
                      '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                      '--file',           const.CONF_FILE_NAME,
                      '--hadoop-arg',     '-libjars',
                      '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                      '--python-archive', 'target/hadoop_ml.tar.gz',
                      '--strict-protocols'])

            with pre_processor_job.make_runner() as runner:
                print('Running pre-processing job...')
                runner.run()
                print('...Done.\n')
                # get pre-processing results and update data handler
                # (!) assuming there is only a single result
                _, encoded_stats = pre_processor_job.parse_output_line(runner.stream_output().next())
                stats = self.data_handler.get_new_statistics().decode(encoded_stats)
                self.conf[const.DATA_HANDLER].set_statistics(stats)
                # overwrite old configuration for training
                serialization.save_object(const.CONF_FILE_NAME, self.conf)


        # ----------- Training ------------------------------------------------

        training_job = TrainingJob(args=[
                  self.data_file,
                  '-r',               self.run_type,
                  '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--file',           const.CONF_FILE_NAME,
                  '--hadoop-arg',     '-libjars',
                  '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--python-archive', 'target/hadoop_ml.tar.gz',
                  '--strict-protocols'])

        with training_job.make_runner() as runner:
            print('Running training job...') 
            runner.run()
            print('...Done.\n')
            # (!) assuming there is only a single result
            _, trained_alg = training_job.parse_output_line(runner.stream_output().next())
        
        # TODO: cleanup pkl files
        
        return self.alg_factory.decode([trained_alg])[0]


    def validate(self, alg, validator):
        '''
        @param alg: Trained algorithm
        @param validator: Validator which should be used for validating a trained model.
        '''
        validation_objects = { const.DATA_HANDLER: self.data_handler, const.TRAINED_ALG: alg, const.VALIDATOR: validator }
        serialization.save_object(const.CONF_FILE_NAME, validation_objects)

        validation_job = ValidationJob(args=[
                  self.data_file,
                  '-r',               self.run_type,
                  '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--file',           const.CONF_FILE_NAME,
                  '--hadoop-arg',     '-libjars',
                  '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--python-archive', 'target/hadoop_ml.tar.gz',
                  '--strict-protocols'])

        print('Running validation job...') 
        with validation_job.make_runner() as runner:
            runner.run()
            print('...Done.\n')
            # (!) assuming there is only a single result
            _, stats = validation_job.parse_output_line(runner.stream_output().next())
        return stats


    def save_data_handler(self, file_name):
        serialization.save_object(file_name, self.data_handler)


def class_from_name(name):
    return getattr(sys.modules[__name__], name)
