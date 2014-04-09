from mrjob.job import MRJob
import sys
from jobs.pre_processor_job import PreProcessorJob
from jobs.validation_job import ValidationJob
from utils import serialization
from jobs.training_job import TrainingJob
import constants.internal as const
from constants import run_type


class Engine(MRJob):

    def __init__(self, alg_factory, data_file, data_handler=None, data_handler_file=None, ml_lib_jar='libs/ml-hadoop-lib.jar', strict_protocols=True, verbose=False):
        '''
        @param alg_factory: Algorithm factory implementing AbstractAlgorithmFactory
        @param data_file: Path (string) to data (either local or HDFS)
        @param data_handler: Optional: Data handler implementing AbstractDataHandler
        @param data_handler_file: Optional: If set the data handler is loaded
        from the file with the given file name omitting pre-processing.
        @param ml_lib_jar: Optional: Location where to find the 
        "ml-hadoop-lib.jar". Default is 'libs/ml-hadoop-lib.jar'.
        @param strict_protocols: Optional: Specifies if an exception should be
        raised if a line can not be parsed from the input protocol.
        Default is True.
        @param verbose: Switches verbose output on/off. Default is False.
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
        self.strict_protocols = strict_protocols
        self.ml_lib_jar = ml_lib_jar
        self.ml_lib_jar_name = self.ml_lib_jar[self.ml_lib_jar.rfind('/')+1:]
        self.verbose = verbose


    def _create_job_args(self, _run_type):
        '''
        Creates a list containing all arguments to configure mr job jobs.
        @param _run_type: Specifies how the job should be run.
        '''
        job_args = [                      self.data_file,
                                    '-r', _run_type,
                                '--file', const.CONF_FILE_NAME,
                      '--python-archive', 'target/hadoop_ml.tar.gz'
                     ]
        # add java library depending on running on EMR or Hadoop
        if _run_type == run_type.EMR:
#             job_args.extend([ '--file', self.ml_lib_jar + '#/home/hadoop/lib' ])
            job_args.extend([ '--bootstrap-cmd', '\'${HADOOP_HOME}/bin/hadoop distcp s3n://mybucket/ml-hadoop-lib.jar /home/hadoop/lib\'' ])
            job_args.extend([ '--hadoop-arg', '-libjars', '--hadoop-arg', self.ml_lib_jar_name ])
        elif _run_type == run_type.HADOOP:
            job_args.extend([ '--hadoop-arg', '-libjars', '--hadoop-arg', self.ml_lib_jar ])

        if self.strict_protocols:
            job_args.append('--strict-protocols')
        return job_args
    
    
    def start(self, _run_type=run_type.HADOOP):
        '''
        Starts this engine performing pre-processing and training. If the engine
        was initialized with an already existing data handler pre-processing is
        skipped.
        @param _run_type: Optional: Specifies how to run pre-processing and 
        training. Default is Hadoop.
        '''

        # ---------- Setup ----------------------------------------------------

        job_args = self._create_job_args(_run_type)

        # serialize data handler and factory
        print('Serializing data handler and algorithm factory...')
        self.conf = { const.DATA_HANDLER: self.data_handler, const.ALG_FACTORY: self.alg_factory }
        serialization.save_object(const.CONF_FILE_NAME, self.conf)
        print('...Done.\n')


        # -------- Pre-Processing ---------------------------------------------

        if self.skip_pre_processing:
            print("Skipping pre-processing\n")
        else:
            pre_processor_job = PreProcessorJob(args=job_args)
            pre_processor_job.set_up_logging(verbose=self.verbose, stream=sys.stdout)

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

        training_job = TrainingJob(args=job_args)

        with training_job.make_runner() as runner:
            print('Running training job...') 
            runner.run()
            print('...Done.\n')
            # (!) assuming there is only a single result
            _, trained_alg = training_job.parse_output_line(runner.stream_output().next())
        
        # TODO: cleanup pkl files
        
        return self.alg_factory.decode([trained_alg])[0]


    def validate(self, alg, validator, _run_type=run_type.HADOOP):
        '''
        @param alg: Trained algorithm
        @param validator: Validator which should be used for validating the 
        trained model.
        @param _run_type: Optional: Specifies how to run the validation. 
        Default is Hadoop
        '''
        validation_objects = { const.DATA_HANDLER: self.data_handler, const.TRAINED_ALG: alg, const.VALIDATOR: validator }
        serialization.save_object(const.CONF_FILE_NAME, validation_objects)
        job_args = self._create_job_args(_run_type)
        validation_job = ValidationJob(args=job_args)

        print('Running validation job...') 
        with validation_job.make_runner() as runner:
            runner.run()
            print('...Done.\n')
            # (!) assuming there is only a single result
            _, stats = validation_job.parse_output_line(runner.stream_output().next())
        return stats


def class_from_name(name):
    return getattr(sys.modules[__name__], name)
