import unittest
from reader.CSVReader import CSVReader


class CSVReaderTest(unittest.TestCase):
    
# TODO    
#     def setUp(self):
#         self.reader = CSVReader()
#         self.reader.set_data_source('../../../../data/wine-quality/winequality-red.csv')
#     
#     def testReadCSVFile(self):
#         data = self.reader.get_data()
#         assert data is not None
#         
#     def testReadCSVFileFail(self):
#         with self.assertRaises(Exception) as cm:
#             self.reader.set_data_source('')
#             self.reader.get_data()
#         
#         assert cm.exception is not None
        
if __name__ == '__main__':
    unittest.main()