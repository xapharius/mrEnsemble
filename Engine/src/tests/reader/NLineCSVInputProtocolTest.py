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
        for _ in range(10):
            size = np.random.random_integers(100, 1000, 1)[0]
            input_str = []
            expected = []
            for _ in range(size):
                numbers = np.random.random_integers(0, 10000, 100)
                numbers_str = map(str, numbers)
                numbers_str = ', '.join(numbers_str)
                input_str.append(numbers_str)
                expected.append(numbers)
            input_str = '0\t' + '\\n'.join(input_str)
            _, read = self.protocol.read(input_str)
            assert np.array_equal(read, np.array(expected))
    
    def test_write_single_line(self):
        data = np.array([[1, 2, 3]])
        assert '0\t1,2,3' == self.protocol.write(0, data)
    
    def test_write_multiple_lines(self):
        for _ in range(10):
            size = np.random.random_integers(100, 1000, 1)[0]
            expected_str = []
            input_arr = []
            for _ in range(size):
                numbers = np.random.random_integers(0, 10000, 100)
                numbers_str = map(str, numbers)
                numbers_str = ','.join(numbers_str)
                expected_str.append(numbers_str)
                input_arr.append(numbers)
            input_arr = np.array(input_arr)
            expected_str = '0\t' + '\\n'.join(expected_str)
            assert expected_str == self.protocol.write(0, input_arr)
    
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