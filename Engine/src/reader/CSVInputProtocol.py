class CSVInputProtocol():
    '''
    MrJob input protocol that creates a list from a CSV line
    '''
    
    def read(self, line):
        values = line.decode('utf-8').split(',')
        values = [x.strip() for x in values]
        return values
    
    def write(self, key, value):
        return bytearray(','.join(value), 'utf-8')
