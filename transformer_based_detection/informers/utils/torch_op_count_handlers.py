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

def add_sub_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:
    input_dims = [get_dims(v) for v in inputs]

    # print(f'Input dims: {input_dims}')

    summand_0_dim = input_dims[0]
    summand_1_dim = input_dims[1]

    # print(len(summand_0_dim))
    # print(len(summand_0_dim))

    if not len(summand_0_dim) and\
                not len(summand_1_dim):
        return 2

    max_dims = [max(dim_summand_0, dim_summand_1)\
                    for dim_summand_0, dim_summand_1\
                        in zip(summand_0_dim, summand_1_dim)]
    
    # print(f'Max dims: {max_dims}')

    return prod(max_dims)



