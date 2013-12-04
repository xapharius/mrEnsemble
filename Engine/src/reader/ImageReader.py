from reader import BaseReader

class ImageReader(BaseReader):
    
    def get_input_format(self):
        # TODO implement WholeFileInputFormat
        return 'org.apache.hadoop.mapred.WholeFileInputFormat'