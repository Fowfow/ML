#!/usr/bin/python3

def factorial(n):
    """
    >>> [factorial(n) for n in range(6)]
    [1, 1, 2, 6, 24, 120]
    >>> factorial(-1)
    Traceback (most recent call last):
        ...
    ValueError: n must be >= 0.
    >>> factorial(30)
    265252859812191058636308480000000
    >>> factorial(30.0)
    2.6525285981219103e+32
    >>> factorial(30.1)
    Traceback (most recent call last):
        ...
    ValueError: n must be an exact integer.
    >>> factorial(1e100)
    Traceback (most recent call last):
        ...
    OverflowError: n is too large.
    """
    import math

    if (n < 0):
        raise ValueError("n must be >= 0.")
    if (math.floor(n) != n):
        raise ValueError("n must be an exact integer.")    
    if (n + 1 == n):
        raise OverflowError("n is too large.")
            
    return 1 if n ==0 else n*factorial(n-1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()            