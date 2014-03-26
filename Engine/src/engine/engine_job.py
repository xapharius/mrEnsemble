import mrjob
from mrjob.job import MRJob

from utils import serialization
import constants as const


class EngineJob(MRJob):
    '''
    Base class for all M/R Jobs of this ML engine.
    '''

    def init(self):
        try:
            if self.is_initialized:
                return
        except AttributeError:
            # assuming configuration is a dictionary
            conf = serialization.load_object('configuration.pkl')
            # create attribute for each object in the loaded configuration
            # data_handler always has to be present in configuration
            for key, value in conf.iteritems():
                setattr(self, key, value)
            self.pre_processor = self.data_handler.get_pre_processor()
            self.data_processor = self.data_handler.get_data_processor()
            self.data_conf = self.data_handler.get_configuration()
            self.is_initialized = True

    def get_data_handler(self):
        return getattr(self, const.DATA_HANDLER)

    def get_pre_processor(self):
        return self.get_data_handler().get_pre_processor()

    def get_data_processor(self):
        return self.get_data_handler().get_data_processor()

    def get_statistics(self):
        return self.get_data_handler().get_statistics()

    def get_alg_factory(self):
        return getattr(self, const.ALG_FACTORY)

    def get_validator(self):
        return getattr(self, const.VALIDATOR)

    def get_trained_alg(self):
        return getattr(self, const.TRAINED_ALG)


    def steps(self):
        return [
            self.mr( mapper_init  = self.init,
                     mapper       = self.mapper,
                     reducer_init = self.init,
                     reducer      = self.reducer )]


    def input_protocol(self):
        try:
            input_protocol = self.data_conf.get_input_protocol()
        except AttributeError:
            self.init()
            input_protocol = self.data_conf.get_input_protocol()
        return input_protocol()

    def internal_protocol(self):
        try:
            internal_protocol = self.data_conf.get_internal_protocol()
        except AttributeError:
            self.init()
            internal_protocol = self.data_conf.get_internal_protocol()
        return internal_protocol()

    def output_protocol(self):
        try:
            output_protocol = self.data_conf.get_output_protocol()
        except AttributeError:
            self.init()
            output_protocol = self.data_conf.get_output_protocol()
        return output_protocol()


    def hadoop_input_format(self):
        try:
            input_format = self.data_conf.get_hadoop_input_format()
        except AttributeError:
            self.init()
            input_format = self.data_conf.get_hadoop_input_format()
        return input_format


    def jobconf(self):
        try:
            custom_jobconf = self.data_conf.get_job_conf()
        except AttributeError:
            self.init()
            custom_jobconf = self.data_conf.get_job_conf()
        orig_jobconf = super(EngineJob, self).jobconf()
        return mrjob.conf.combine_dicts(orig_jobconf, custom_jobconf)
