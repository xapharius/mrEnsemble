'''
Created on Apr 8, 2014

@author: Simon
'''
import unittest
from protocol.n_image_input_protocol import NImageInputProtocol
from skimage import io as skio
import struct
from numpy.testing.utils import assert_array_equal
from numpy.ma.testutils import assert_equal
import io
from encodings.base64_codec import base64_encode
import base64


class NImagesInputProtocolTest(unittest.TestCase):

    def setUp(self):
        self.protocol = NImageInputProtocol()

    def testReadSingleImage(self):
        f = open('../../../../data/test-images/cat_programming.png', "rb")
        
        file_bytes = bytearray(f.read())
        len_bytes = bytearray(struct.pack('>i', len(file_bytes)))
        _, images = self.protocol.read(f.name + '\t' + base64_encode(str(len_bytes + file_bytes))[0])
        
        exp = skio.imread('../../../../data/test-images/cat_programming.png')
        
        assert_array_equal(images[0], exp)
    
    def testWriteSingleImage(self):
        image = skio.imread('../../../../data/test-images/cat_programming.png')
        img_list = [image]
        key = 'cat'
        image_bytes = self.protocol.write(key, img_list)
        
        byte_stream = io.BytesIO()
        skio.imsave(byte_stream, image)
        file_bytes = byte_stream.getvalue()
        byte_stream.close()
        len_bytes = bytearray(struct.pack('>i', len(file_bytes)))
        
        assert_equal( image_bytes, key + '\t' + base64.b64encode(len_bytes + file_bytes) )

    def testReadWriteReadSingleImage(self):
        f = open('../../../../data/test-images/cat_programming.png', "rb")
        file_bytes = bytearray(f.read())
        len_bytes = bytearray(struct.pack('>i', len(file_bytes)))
        
        exp = skio.imread('../../../../data/test-images/cat_programming.png')
        key = 'cat'
        key, images = self.protocol.read(key + '\t' + base64.b64encode(str(len_bytes + file_bytes)))
        image_bytes = self.protocol.write(key, images)
        _, images = self.protocol.read(image_bytes)
        
        assert_array_equal(images[0], exp)
    
    def testReadMultipleImages(self):
        f = open('../../../../data/test-images/cat_programming.png', "rb")
        file_bytes = bytearray(f.read())
        len_bytes = bytearray(struct.pack('>i', len(file_bytes)))
        # five times the same image
        _, images = self.protocol.read(f.name + '\t' + base64_encode(str(len_bytes + file_bytes))[0]*5)
        
        exp = skio.imread('../../../../data/test-images/cat_programming.png')
        
        for img in images:
            assert_array_equal(img, exp)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testReadSingleImage']
    unittest.main()