from reader.BaseReader import BaseReader
from numpy import genfromtxt

class CSVReader(BaseReader):
    
    def get_data(self):
        if self.data is None:
            raise Exception('No data source has been set!')
        return self.data
    
    def set_data_source(self, csv_source):
        self.data = genfromtxt(csv_source, delimiter=',', skip_header=1)
