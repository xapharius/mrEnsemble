from mrjob.job import MRJob
from engine_job import EngineJob


class Engine(MRJob):

    def __init__(self, alg_factory, data_handler, data_file):
        self.alg_factory = alg_factory
        self.data_handler = data_handler
        self.data_file = data_file
    
    def start(self):
        # TODO: serialize configuration and upload it too
        job = EngineJob(args=['-r', 'hadoop', self.data_file, '--file', '../HadoopLib/target/ml-hadoop-lib.jar', '--hadoop-arg', '-libjars', '--hadoop-arg', '../HadoopLib/target/ml-hadoop-lib.jar', '--python-archive', 'target/hadoop_ml.tar.gz', '--strict-protocols'])
        with job.make_runner() as runner:
            runner.run()
            print list(runner.stream_output())