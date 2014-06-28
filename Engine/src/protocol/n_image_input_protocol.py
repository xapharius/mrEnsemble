'''
Created on Apr 9, 2014

@author: Simon
'''
import matplotlib
from utils import logging
# !important: tell matplotlib not to try rendering to a window
matplotlib.use('Agg')
import io
import struct
from skimage import io as skio
import base64


class NImageInputProtocol(object):
    '''
    MrJob input protocol that reads a number of PNG images that are encoded as 
    follows: Base64 of | no. of bytes of image (4 bytes) | image bytes | ...
    The result is a list of numpy arrays.
    The write method encodes a list of images (numpy arrays) as base64 byte 
    string in the same way as the input for the read method.
    '''


    def read(self, data):
        key, enc_value = data.split('\t', 1)
        value = base64.b64decode(enc_value)
        pos = 0
        image_arrs = []
        logging.info('decoded number of bytes: ' + str(len(value)))
        while pos < len(value):
            image_len = struct.unpack('>i', value[pos:pos+4])[0]
            pos += 4
            logging.info('reading image of length: ' + str(image_len) + '\n')
            image_arr = skio.imread(io.BytesIO(value[pos:pos + image_len]))
            logging.info('done reading')
            image_arrs.append(image_arr)
            pos += image_len
        logging.info('Got ' + str(len(image_arrs)) + ' images')
        return key, image_arrs

    def write(self, key, img_list):
        logging.info('Writing ' + str(len(img_list)) + ' images')
        byte_stream = io.BytesIO()
        for img in img_list:
            # get image bytes
            temp_stream = io.BytesIO()
            skio.imsave(temp_stream, img)
            img_bytes = temp_stream.getvalue()
            temp_stream.close()
            
            # get length of bytes in four bytes
            img_len = len(img_bytes)
            logging.info('Writing image of length ' + str(img_len))
            len_bytes = bytearray(struct.pack('>i', img_len))
            
            # save length and image bytes to the result
            byte_stream.write(str(len_bytes))
            byte_stream.write(img_bytes)
        
        final_bytes = byte_stream.getvalue()
        byte_stream.close()
        encoded = base64.b64encode(final_bytes)
        logging.info('Done writing. Final number of bytes: ' + str(len(final_bytes)))
        return '%s\t%s' % (key, encoded)
