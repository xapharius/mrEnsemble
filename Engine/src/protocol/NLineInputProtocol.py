import sys

class NLineInputProtocol(object):
    '''
    classdocs
    '''

    def read(self, line):
        sys.stderr.write('input: "' + line + '"\n')
        k_str, v_str = line.split('\t', 1)
        lines = v_str.split('\\n')
        return long(k_str), lines
    
    def write(self, key, value):
        return '%s\t%s' % (key, value)