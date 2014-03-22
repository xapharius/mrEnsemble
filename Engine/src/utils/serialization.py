'''
Created on Mar 22, 2014

@author: Simon
'''
import pickle

def load_object(file_name):
    pkl_file = open(file_name, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()
    return obj

def save_object(file_name, obj):
    output = open(file_name, 'wb')
    # use highest protocol version available
    pickle.dump(obj, output, -1)
    output.close()