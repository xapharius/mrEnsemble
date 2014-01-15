import unittest
from reader.NLineCSVInputProtocol import NLineCSVInputProtocol
import numpy as np

class CSVInputProtocolTest(unittest.TestCase):
    
    
    def setUp(self):
        self.protocol = NLineCSVInputProtocol()
    
    def test_read_single_line(self):
        _, read = self.protocol.read('0\t1, 2, 3')
        assert np.array_equal(read, np.array([[1, 2, 3]]))
    
    def test_read_multiple_lines(self):
        _, read = self.protocol.read('0\t1, 2, 3\\n4, 5, 6')
        assert np.array_equal(read, np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_write_single_line(self):
        data = np.array([[1, 2, 3]])
        assert '0\t1,2,3' == self.protocol.write(0, data)
    
    def test_write_multiple_lines(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        assert '0\t1,2,3\\n4,5,6' == self.protocol.write(0, data)
    
    def test_read_write_single_line(self):
        data = '0\t1.0,2.0,3.0'
        key, read = self.protocol.read(data)
        assert data == self.protocol.write(key, read)
        
    def test_read_write_multiple_lines(self):
        data = '0\t1.0,2.0,3.0\\n4.0,5.0,6.0'
        key, read = self.protocol.read(data)
        assert data == self.protocol.write(key, read)


if __name__ == '__main__':
    unittest.main()