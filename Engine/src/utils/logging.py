import sys
from time import gmtime, strftime

def log(type_str, msg):
    time_str = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    sys.stderr.write(time_str + ' ' + type_str + ': ' + msg + '\n')

def info(msg):
    log('INFO', msg)

def warn(msg):
    log('WARNING', msg)

def error(msg):
    log('ERROR', msg)