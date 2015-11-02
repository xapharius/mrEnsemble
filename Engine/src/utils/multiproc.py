'''
Created on Aug 4, 2015

@author: xapharius
'''
import numpy as np

def parallel_function(f):
    def easy_parallize(f, sequence):
        """ assumes f takes sequence as input, easy w/ Python's scope """
        from multiprocessing import Pool
        pool = Pool(processes=3)  # depends on available cores
        result = pool.map(f, sequence)  # for i in sequence: result[i] = f(i)
        cleaned = [x for x in result if x is not None]  # getting results
        cleaned = np.asarray(cleaned)
        pool.close()  # not optimal! but easy
        pool.join()
        return cleaned
    from functools import partial
    return partial(easy_parallize, f)