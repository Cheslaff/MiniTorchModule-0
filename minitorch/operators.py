"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x1: float, x2: float) -> float:
    """Returns x1 * x2"""
    return x1 * x2


def id(x: float) -> float:
    """Returns x untouched"""
    return x


def add(x1: float, x2: float) -> float:
    """Returns x1 + x2"""
    return x1 + x2


def neg(x: float) -> float:
    """Returns x multiplied by -1"""
    return -x


def lt(x1: float, x2: float) -> bool:
    """Returns True if x1 < x2 else False"""
    return x1 < x2


def eq(x1: float, x2: float) -> bool:
    """Returns True if x1==x2 else False"""
    return x1 == x2


def max(x1: float, x2: float) -> float:
    """Returns maximum from x1 and x2"""
    return x1 if x1 > x2 else x2


def is_close(x1: float, x2: float) -> float:
    """Checks whether two numbers are close |x1-x2| < 1e-2"""
    return abs(x1 - x2) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid Activation Function: https://en.wikipedia.org/wiki/Sigmoid_function"""
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    """ReLU Activation Function: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)"""
    return x if x > 0 else 0


def log(x: float) -> float:
    """Logarithm of value x"""
    return math.log(x)


def exp(x: float) -> float:
    """Exponentiation of value x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Inverse of a scalar value"""
    return 1 / x


def log_back(x1: float, x2: float) -> float:
    """Backward pass for log multiplied by 2nd variable"""
    return inv(x1) * x2


def inv_back(x1: float, x2: float) -> float:
    """Backward pass for inv multiplied by 2nd variable"""
    return -1 / x1**2 * x2


def relu_back(x1: float, x2: float) -> float:
    """Backward pass for relu multiplied by 2nd variable"""
    return (x1 > 0) * x2


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(
    function: Callable[[float], float],
) -> Callable[[Iterable[float]], Iterable[float]]:
    """Maps given function to each element of Iterable

    Arguments:
    ---------
        function - function being applied to Iterable
    Returns:
        function "inner" - function, which applies function(argument) to array (argument)

    """

    def inner(array: Iterable[float]) -> Iterable[float]:
        result = []
        for item in array:
            result.append(function(item))
        return result

    return inner


def zipWith(
    zipper: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Zips 2 Iterables using given function

    Arguments:
    ---------
        zipper - function applied to element couples to combine them
    Returns:
        function "inner" - function, which applies zipper to couples of arrays

    """

    def inner(array1: Iterable[float], array2: Iterable[float]) -> Iterable[float]:
        result = []
        for item1, item2 in zip(array1, array2):
            result.append(zipper(item1, item2))
        return result

    return inner


def reduce(
    reducer: Callable[[float, float], float],
) -> Callable[[Iterable[float]], float]:
    """Reduces Iterable to one scalar using reducer function

    Arguments:
    ---------
        reducer - function being used to manipulate scalars
    Returns:
        function "inner" - function, which applies reducer to array (argument)

    """

    def inner(array: Iterable[float]) -> float:
        result = 0.0
        for i, item in enumerate(array):
            if i == 0:
                result = item
            else:
                result = reducer(result, item)
        return result

    return inner


# Special functions

negList = map(neg)
addLists = zipWith(add)
sum = reduce(add)
prod = reduce(lambda x, y: x * y)
