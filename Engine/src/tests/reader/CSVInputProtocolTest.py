import unittest
from reader.CSVInputProtocol import CSVInputProtocol
import numpy as np

class CSVInputProtocolTest(unittest.TestCase):
    
    
    def setUp(self):
        self.protocol = CSVInputProtocol()
    
    def test_read_single_line(self):
        read = self.protocol.read(bytearray('1, 2, 3', 'utf-8'))
        assert np.array_equal(read, np.array([[1, 2, 3]]))
    
    def test_read_multiple_lines(self):
        read = self.protocol.read(bytearray('1, 2, 3\n4, 5, 6', 'utf-8'))
        assert np.array_equal(read, np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_write_single_line(self):
        data = np.array([[1, 2, 3]])
        assert bytearray('1,2,3', 'utf-8') == self.protocol.write(None, data)
    
    def test_write_multiple_lines(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        assert bytearray('1,2,3\n4,5,6', 'utf-8') == self.protocol.write(None, data)
    
    def test_read_write_single_line(self):
        byte_data = bytearray('1.0,2.0,3.0', 'utf-8')
        read = self.protocol.read(byte_data)
        assert byte_data == self.protocol.write(None, read)
        
    def test_read_write_multiple_lines(self):
        byte_data = bytearray('1.0,2.0,3.0\n4.0,5.0,6.0', 'utf-8')
        read = self.protocol.read(byte_data)
        assert byte_data == self.protocol.write(None, read)


if __name__ == '__main__':
    unittest.main()