import numpy as np
import math

def create_interval_positive(x, adjust_mag=0, verbose=False):
    '''
    Creates a interval of positive numbers around a given number x
    Parameters:
    -----------
    x: a given real, non-zero and positive number
        type: integer, float
    adjust_mag: value by which the order of magnitude of x will be adjusted 
                (i.e. add or substract from order of magnitude)
                type: integer
    verbose: print additional detail 
        type: bool
    Returns:
    --------
    interval: a list of number around the x 
    '''
    assert x > 0, 'Given number x is not a positive and non-zero number'

    x = abs(x)
    order_mag = int(math.floor(math.log(x,10)))

    order_mag += adjust_mag
    if verbose: print('Order of magnitude:',order_mag)
    plus = x + 10*(10**order_mag)
    if verbose: print(plus)

    minus = x - 10*(10**order_mag)
    if verbose: print(minus)
    interval = np.arange(minus,plus+10**(order_mag-1),10**order_mag)
    for num in interval:
        interval = np.delete(interval,np.where(interval<=0))
    if verbose: print(np.round(interval,abs(order_mag)+1))
    return np.round(interval,abs(order_mag)+1)

def create_interval(x, adjust_mag=0, verbose=False):
    '''
    Creates a interval around a given number x
    Parameters:
    -----------
    x: a given real number
        type: integer, float
    adjust_mag: value by which the order of magnitude of x will be adjusted 
                (i.e. add or substract from order of magnitude)
                type: integer
    verbose: print additional detail
        type: bool

    Returns:
    --------
    interval: a list of number around the x 
    '''
    negative = False
    
    if x == 0:
        order_mag = 0
    else:
        if x < 0:
            negative = True
        x = abs(x)
        order_mag = int(math.floor(math.log(x,10)))
    
    order_mag += adjust_mag
    if verbose: print('Order of magnitude:',order_mag)
    plus = x + 10*(10**order_mag)
    if verbose: print(plus)

    minus = x - 10*(10**order_mag)
    if verbose: print(minus)
    interval = np.arange(minus,plus+10**(order_mag-1),10**order_mag)
    if negative == True:
        interval = -1*interval
    if verbose: print(np.round(interval,abs(order_mag)+1))

    return np.round(interval,abs(order_mag)+1)
