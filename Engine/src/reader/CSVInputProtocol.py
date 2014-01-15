import numpy as np

class CSVInputProtocol():
    '''
    MrJob input protocol that creates a matrix from multiple CSV lines parsing
    each value as float
    '''
    
    def read(self, line_bytes):
        lines = np.array([line.split(',') for line in line_bytes.decode('utf-8').split('\n')])
        strip = np.vectorize(lambda cell: float(cell.strip()))
        return strip(lines);
    
    def write(self, key, value):
        to_str = np.vectorize(lambda cell: str(cell))
        value = to_str(value)
        lines = [','.join(line) for line in value]
        return bytearray('\n'.join(lines), 'utf-8')
