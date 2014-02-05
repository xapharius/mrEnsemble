import numpy as np
import sys

class NLineCSVInputProtocol():
    '''
    MrJob input protocol that creates a matrix from multiple CSV lines parsing
    each value as float
    '''
    
    def read(self, data):
        # remove trailing line delimiter
        if data.endswith('\\n'):
            data = data[0:len(data)-2]
        key, value = data.split('\t', 1)
        delimiter = ','
        if value.find(';') != -1:
            delimiter = ';'
        lines = value.split('\\n')
        
        # TODO: sometimes a line is corrupted, although it is correctly built
        # by java record reader
        # hack for messed up lines
        arr = []
        len_line = len(lines[0].split(delimiter))
        for line in lines:
            temp_line = line.split(delimiter)
            if len(temp_line) == len_line:
                arr.append(temp_line)
            else:
                sys.stderr.write('discarding line with wrong length:"' + line + '" \n')
        
        # convert to numpy array
        arr = np.array(arr)
        sys.stderr.write('shape: ' + str(arr.shape) + '\n')
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