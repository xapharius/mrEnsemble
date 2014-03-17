'''
Created on Jan 15, 2014

@author: Simon
'''
from datahandler.AbstractConfiguration import AbstractConfiguration
import reader
import mrjob.protocol

class NumericalConfiguration(AbstractConfiguration):
    '''
    classdocs
    '''

    INPUT_PROTOCOL = reader.NLineCSVInputProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.NLineFileInputFormat'

    def __init__(self, lines_per_map):
        self.lines_per_map = lines_per_map

    def get_input_protocol(self):
        return self.INPUT_PROTOCOL

    def get_internal_protocol(self):
        return self.INTERNAL_PROTOCOL
    
    def get_output_protocol(self):
        return self.OUTPUT_PROTOCOL

    def get_hadoop_input_format(self):
        return self.HADOOP_INPUT_FORMAT

    def get_job_conf(self):
        return { 'hadoopml.fileinput.linespermap': self.lines_per_map }
