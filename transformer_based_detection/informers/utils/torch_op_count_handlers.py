import typing
from collections import Counter, OrderedDict
from numbers import Number
from typing import Any, Callable, List, Optional, Union

import numpy as np

try:
    from math import prod
except ImportError:
    from numpy import prod

Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]


def get_dims(val: Any) -> Optional[List[int]]:
    """
    Get the dims from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None


def add_sub_mul_div_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:

    # for v in inputs:
    #     if v.isCompleteTensor():
    #         print(v.type().dtype())

    input_dims = [get_dims(v) for v in inputs]

    # print(f'Input dims: {input_dims}')

    input_0_dims = input_dims[0]
    input_1_dims = input_dims[1]

    # print(len(input_0_dims))
    # print(len(input_0_dims))

    if not len(input_1_dims): 
        if not len(input_0_dims):
            return 2
        else:
            return prod(input_0_dims)

    max_dims = [max(input_0_dim, input_1_dim)\
                    for input_0_dim, input_1_dim\
                        in zip(input_0_dims, input_1_dims)]
    
    # print(f'Max dims: {max_dims}')

    return prod(max_dims)


def sum_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:

    # for v in inputs:
    #     if v.isCompleteTensor():
    #         print(v.type().dtype())

    input_dims = [get_dims(v) for v in inputs]

    # print(f'Input dims: {input_dims}')

    output_dims = [get_dims(v) for v in outputs]

    # print(f'Output dims: {output_dims}')

    ops = ((prod(input_dims[0])//\
                prod(output_dims[0])) - 1)*\
                prod(input_dims[0])

    return ops


def mean_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:
    return sum_op_handler(inputs, outputs) + 1


def cumsum_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:
    
    # Note: This implementation is only valid for the argument dim=-2,
    # as used in the function _get_initial_context of the class ProbAttention
    
    for v in inputs:
        if v.isCompleteTensor():
            print(dir(v.type()))

    input_dims = [get_dims(v) for v in inputs]

    print(f'Input dims: {input_dims}')

    output_dims = [get_dims(v) for v in outputs]

    print(f'Output dims: {output_dims}')

    return 5