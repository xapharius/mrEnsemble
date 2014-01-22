import numpy as np
import sys

class NLineCSVInputProtocol():
    '''
    MrJob input protocol that creates a matrix from multiple CSV lines parsing
    each value as float
    '''
    
    def read(self, data):
        key, value = data.split('\t', 1)
        delimiter = ','
        if value.find(';') != -1:
            delimiter = ';'
        lines = value.split('\\n')
        arr = np.array([line.split(delimiter) for line in lines])
        sys.stderr.write('arr: ' + str(arr) + '\n')
        result = np.zeros(arr.shape)
        # strip and convert values to float
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i,j] = float(arr[i,j].strip())
                
        return key, result;
    
    def write(self, key, value):
        to_str = np.vectorize(lambda cell: str(cell))
        value = to_str(value)
        lines = [','.join(line) for line in value]
        return '%s\t%s' % (key, '\\n'.join(lines))
