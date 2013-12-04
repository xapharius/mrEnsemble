import unittest
from reader.CSVInputProtocol import CSVInputProtocol


class CSVInputProtocolTest(unittest.TestCase):
    
    
    def setUp(self):
        self.protocol = CSVInputProtocol()
    
    def testRead(self):
        read = self.protocol.read(bytearray(' a, b, c', 'utf-8'))
        assert read == ['a', 'b', 'c']
    
    def testWrite(self):
        data = ['a', 'b', 'c']
        assert bytearray(','.join(data), 'utf-8') == self.protocol.write(None, data)
    
    def testReadWrite(self):
        byte_data = bytearray('a,b,c', 'utf-8')
        read = self.protocol.read(byte_data)
        assert byte_data == self.protocol.write(None, read)


if __name__ == '__main__':
    unittest.main()