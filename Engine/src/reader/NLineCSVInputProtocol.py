import numpy as np

class NLineCSVInputProtocol():
    '''
    MrJob input protocol that creates a matrix from multiple CSV lines parsing
    each value as float
    '''
    
    def read(self, data):
        key, value = data.split('\t', 1)
        lines = value.split('\\n')
        arr = np.array([line.split(',') for line in lines])
        strip = np.vectorize(lambda cell: float(cell.strip()))
        return key, strip(arr);
    
    def write(self, key, value):
        to_str = np.vectorize(lambda cell: str(cell))
        value = to_str(value)
        lines = [','.join(line) for line in value]
        return '%s\t%s' % (key, '\\n'.join(lines))
