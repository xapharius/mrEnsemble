import mrjob

class BaseReader(object):

    def get_input_format(self):
        return 'org.apache.hadoop.mapred.TextInputFormat'

    def get_input_protocol(self):
        return mrjob.protocol.RawValueProtocol

    def get_internal_protocol(self):
        return mrjob.protocol.JSONProtocol
    
    def get_outut_protocol(self):
        return mrjob.protocol.JSONProtocol