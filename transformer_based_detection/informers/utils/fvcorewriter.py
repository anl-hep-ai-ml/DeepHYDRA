from __future__ import annotations

import re
import os
import typing
from collections import Counter, OrderedDict
from numbers import Number
from typing import Any, List, Optional, Union, Tuple
import unicodedata
import json

import numpy as np
from torch import Tensor
from torch import nn
from torch import finfo, iinfo, is_floating_point
from torch import dtype as tdt
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

try:
    from math import prod
except ImportError:
    from numpy import prod


def _get_dims(val: Any) -> Optional[List[int]]:
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


def _add_sub_mul_div_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:

    input_dims = [_get_dims(v) for v in inputs]

    input_0_dims = input_dims[0]
    input_1_dims = input_dims[1]

    if not len(input_1_dims): 
        if not len(input_0_dims):
            return 2
        else:
            return prod(input_0_dims)

    max_dims = [max(input_0_dim, input_1_dim)\
                    for input_0_dim, input_1_dim\
                        in zip(input_0_dims, input_1_dims)]

    return prod(max_dims)


def _sum_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:

    input_dims = [_get_dims(v) for v in inputs]
    output_dims = [_get_dims(v) for v in outputs]

    ops = int(((prod(input_dims[0])//\
                    prod(output_dims[0])) - 1)*\
                    prod(input_dims[0]))

    return ops


def _mean_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:
    return _sum_op_handler(inputs, outputs) + 1


def _cumsum_op_handler(inputs: List[Any], outputs: List[Any]) -> Number:
    
    # Note: This implementation is only valid for the argument dim=-2,
    # as used in the function _get_initial_context of the class ProbAttention

    input_dims = [_get_dims(v) for v in inputs]

    b, m, n, k = input_dims[0]

    ops = int(b*m*k*n*(n - 1)/2)

    return ops


class FVCoreWriter():

    def __init__(self,
                    model: nn.Module | None = None,
                    inputs: Union[Tensor, Tuple[Tensor, ...]] | None = None) -> None:
        self._flop_count_analysis = None
        self._activation_count_analysis = None
        self._initialized = False

        if model:
            self.analyze(model,
                            inputs)

    def analyze(self,
                    model: nn.Module,
                    inputs: Union[Tensor, Tuple[Tensor, ...]]) -> None:

        self._flop_count_analysis =\
                    FlopCountAnalysis(model,
                                        inputs)\
                                            .set_op_handle('aten::add', _add_sub_mul_div_op_handler)\
                                            .set_op_handle('aten::sub', _add_sub_mul_div_op_handler)\
                                            .set_op_handle('aten::mul', _add_sub_mul_div_op_handler)\
                                            .set_op_handle('aten::div', _add_sub_mul_div_op_handler)\
                                            .set_op_handle('aten::sum', _sum_op_handler)\
                                            .set_op_handle('aten::mean', _mean_op_handler)\
                                            .set_op_handle('aten::cumsum', _cumsum_op_handler)

        self._activation_count_analysis =\
                    ActivationCountAnalysis(model,
                                            inputs)

        self._initialized = True


    def get_flop_dict(self,
                        by: str = 'by_module') -> dict:

        if not self._initialized:
            raise RuntimeError('FVCoreWriter is not initialized. '
                                'Initialization is performed by passing '
                                'a model to the class in the constructor'
                                'or by calling analyze() after '
                                'construction.')

        if by == 'by_module':
            return dict(self._flop_count_analysis.by_module())
        elif by == 'by_operator':
            return dict(self._flop_count_analysis.by_operator())
        else:
            raise ValueError(f'Ordering {by} is unknown')


    def get_activation_dict(self,
                                by: str = 'by_module') -> dict:

        if not self._initialized:
            raise RuntimeError('FVCoreWriter is not initialized. '
                                'Initialization is performed by passing '
                                'a model to the class in the constructor'
                                'or by calling analyze() after '
                                'construction.')

        if by == 'by_module':
            return dict(self._activation_count_analysis.by_module())
        elif by == 'by_operator':
            return dict(self._activation_count_analysis.by_operator())
        else:
            raise ValueError(f'Ordering {by} is unknown')


    def write_flops_to_json(self,
                                output_filename: str,
                                by: str = 'by_module') -> None:
        self._write_to_json(output_filename,
                                self.get_flop_dict(by))


    def write_activations_to_json(self,
                                    output_filename: str,
                                    by: str = 'by_module') -> None:
        self._write_to_json(output_filename,
                                self.get_activation_dict(by))


    def _write_to_json(self,
                        output_filename: str,
                        data: dict,
                        allow_unicode_filename=False) -> None:

        path, filename = os.path.split(output_filename)

        path = os.path.abspath(path)

        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'[. ]+$', '', filename)

        output_filename = os.path.join(path, filename)

        with open(output_filename, 'w') as output_file:
            json.dump(data, output_file)

