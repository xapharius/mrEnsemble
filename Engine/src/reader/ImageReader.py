from reader import BaseReader
from PIL import Image

class ImageReader(BaseReader):
    
    def get_data(self):
        if self.data is None:
            raise Exception('No data source has been set!')
        return self.data
    
    def set_data_source(self, csv_source):
        self.data = Image.open()

    def send_to_hdfs(self):
        pass

    def get_input_protocol(self):
        pass
    
    def get_input_format(self):
        return 'org.apache.hadoop.mapred.WholeFileInputFormat'