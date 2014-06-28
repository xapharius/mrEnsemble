'''
Created on Jan 15, 2014

@author: Simon
'''
from protocol.n_image_input_protocol import NImageInputProtocol
import mrjob.protocol
from datahandler.abstract_job_conf import AbstractJobConf

class ImagePreProcConf(AbstractJobConf):
    '''
    classdocs
    '''
    
    INPUT_PROTOCOL = NImageInputProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.NWholeFileInputFormat'

    def __init__(self, files_per_map):
        self.files_per_map = files_per_map

    def get_input_protocol(self):
        return self.INPUT_PROTOCOL

    def get_internal_protocol(self):
        return self.INTERNAL_PROTOCOL
    
    def get_output_protocol(self):
        return self.OUTPUT_PROTOCOL

    def get_hadoop_input_format(self):
        return self.HADOOP_INPUT_FORMAT

    def get_job_conf(self):
        return { 'hadoopml.fileinput.filespermap': self.files_per_map }
