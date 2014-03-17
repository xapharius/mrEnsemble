'''
Created on Mar 17, 2014

@author: Simon
'''

def equals_with_tolerance(list1, list2, tolerance):
    if not len(list1) == len(list2):
        return False
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > tolerance:
            return False
    return True