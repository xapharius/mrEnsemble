from mrjob.job import MRJob
from engine_job import EngineJob
from datahandler.pre_processor_job import PreProcessorJob
import sys
import pickle


class Engine(MRJob):

    def __init__(self, alg_factory, data_handler, data_file):
        '''
        alg_factory: algorithm factory implementing AbstractAlgorithmFactory
        data_handler: data handler implementing AbstractDataHandler
        data_file: path (string) to data (either local or HDFS)
        '''
        self.alg_factory = alg_factory
        self.data_handler = data_handler
        self.data_file = data_file

    def start(self):

        # ---------- Setup ----------------------------------------------------

        # serialize data handler and factory
        print('Serializing data handler and algorithm factory...')
        self._save_object('data_handler.pkl', self.data_handler)
        self._save_object('alg_factory.pkl', self.alg_factory)
        print('...Done.')


        # -------- Pre-Processing ---------------------------------------------

        print('Running pre-processing job...')
        pre_processor_job = PreProcessorJob(args=[
                  self.data_file,
                  '-r',               'hadoop',
                  '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--file',           'data_handler.pkl',
                  '--hadoop-arg',     '-libjars',
                  '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--python-archive', 'target/hadoop_ml.tar.gz',
                  '--strict-protocols'])

        # create output protocol instance for output parsing
        output_protocol = pre_processor_job.OUTPUT_PROTOCOL()

        with pre_processor_job.make_runner() as runner:
            runner.run()
            # get pre-processing results and update data handler
            # (!) assuming there is only a single result
            (_, stats) = output_protocol.read(runner.stream_output().next())
            self.data_handler.set_statistics(stats)
            # overwrite old data handler file
            self._save_object('data_handler.pkl', self.data_handler)
        print('...Done.')


        # ----------- Training ------------------------------------------------

        print('Running training job...') 
        engine_job = EngineJob(args=[
                  self.data_file,
                  '-r',               'hadoop',
                  '--file',           '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--file',           'data_handler.pkl',
                  '--file',           'alg_factory.pkl',
                  '--hadoop-arg',     '-libjars',
                  '--hadoop-arg',     '../HadoopLib/target/ml-hadoop-lib.jar',
                  '--python-archive', 'target/hadoop_ml.tar.gz',
                  '--strict-protocols'])
        output_protocol = engine_job.OUTPUT_PROTOCOL()
        with engine_job.make_runner() as runner:
            runner.run()
            # (!) assuming there is only a single result
            (_, trained_alg) = output_protocol.read(runner.stream_output().next())
        print('...Done.')


        # ----------- Validation ----------------------------------------------
        return self.alg_factory.decode([trained_alg])[0]

    def _save_object(self, file_name, obj):
        output = open(file_name, 'wb')
        # use highest protocol version available
        pickle.dump(obj, output, -1)
        output.close()


def class_from_name(name):
    return getattr(sys.modules[__name__], name)
