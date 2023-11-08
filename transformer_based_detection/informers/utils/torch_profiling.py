import typing
from collections import Counter, OrderedDict
from numbers import Number
from typing import Any, Callable, List, Optional, Union

import numpy as np
from torch import finfo, iinfo, is_floating_point
from torch import dtype as tdt

try:
    from math import prod
except ImportError:
    from numpy import prod


Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]


def to_bytes(t: tdt) -> int:
    return finfo(t).bits/8 \
                if is_floating_point \
            else iinfo(t).bits/8


def get_params_hook(module,
                        in_tensor,
                        out_tensor,
                        log_filename: str,
                        detailed: bool = False) -> None:

    if all(False for _ in module.children()):

        named_modules = list(module.named_modules())

        module_name = named_modules[0][0]
        module_params = named_modules[0][1]

        # print(f'{module_name}: {module_params}')
        # print(f'{module_params._get_name()}')

        with open(log_filename, 'a') as log_file:
            for param in module.parameters():
                print(param.data.__repr__())
                # print(param.shape)
                # print(f'{module_params._get_name()}: '
                #             f'{int(param.numel()*to_bytes(param.dtype))}')
                # print(f'{module_params._get_name()}: '
                #             f'{int(param.numel()*to_bytes(param.dtype))}',
                #         file=log_file)


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

    ops = int(((prod(input_dims[0])//\
                    prod(output_dims[0])) - 1)*\
                    prod(input_dims[0]))

    return ops


def mean_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:
    return sum_op_handler(inputs, outputs) + 1


def cumsum_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:
    
    # Note: This implementation is only valid for the argument dim=-2,
    # as used in the function _get_initial_context of the class ProbAttention
    
    # for v in inputs:
    #     if v.isCompleteTensor():
    #         print(dir(v.type()))

    input_dims = [get_dims(v) for v in inputs]

    # print(f'Input dims: {input_dims}')

    b, m, n, k = input_dims[0]

    ops = int(b*m*k*n*(n - 1)/2)

    # print(ops)

    return ops