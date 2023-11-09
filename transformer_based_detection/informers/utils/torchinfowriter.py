# Copyright (c) 2022 Tyler Yep
# Modifications copyright (c) 2023 Computing Systems Group

from __future__ import annotations

import os
import json
from collections import defaultdict
from typing import Any,\
                    Iterable,\
                    Mapping,\
                    Optional,\
                    Sequence,\
                    Union

import numpy as np
import pandas as pd
import torch
from torch import nn
import pydot
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from torchinfo import summary
from bigtree import Node, tree_to_dataframe, tree_to_dot
from bigtree.tree.search import find_child_by_name


INPUT_DATA_TYPE = Union[torch.Tensor, np.ndarray, Sequence[Any], Mapping[str, Any]]
CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]


class TorchinfoWriter():

    def __init__(self,
                    model: nn.Module | None = None,
                    input_size: INPUT_SIZE_TYPE | None = None,
                    input_data: INPUT_DATA_TYPE | None = None,
                    batch_dim: int | None = None,
                    cache_forward_pass: bool | None = None,
                    verbose: int | None = None):

        self._initialized = False

        if model:
            self.update_model(model,
                                input_size,
                                input_data,
                                batch_dim,
                                cache_forward_pass,
                                verbose)

    def update_model(self,
                        model: nn.Module,
                        input_size: INPUT_SIZE_TYPE | None = None,
                        input_data: INPUT_DATA_TYPE | None = None,
                        batch_dim: int | None = None,
                        cache_forward_pass: bool | None = None,
                        verbose: int | None = None):

        self.model = model
        self.input_data = input_data
        self.input_size = input_size
        self.batch_dim = batch_dim
        self.cache_forward_pass = cache_forward_pass
        self.verbose = verbose


    def construct_model_tree(self):

        model_summary = summary(self.model,
                                    input_data=self.input_data,
                                    verbose=self.verbose)

        self._nodes_at_level = defaultdict(list)

        postfixes_at_level = defaultdict(lambda: defaultdict(int))

        for layer_info in model_summary.summary_list:

            name, type_ = self._label_to_row_entries(
                            layer_info.get_layer_name(True, False))

            trainable = True if layer_info.trainable == 'True' else False

            if (not layer_info.is_recursive) and trainable:
                self._nodes_at_level[layer_info.depth].append(
                                Node.from_dict({'name': name,
                                                    'Type': type_,
                                                    'Kernel Size': layer_info.kernel_size,
                                                    'Input Size': layer_info.input_size,
                                                    'Output Size': layer_info.output_size,
                                                    'Parameters': layer_info.num_params,
                                                    'MACs': layer_info.macs}))

                self._nodes_at_level[layer_info.depth][-1].sep = '.'

                if layer_info.depth > 0:

                    # Check if a child with the same path as the
                    # element to be inserted exists
                    if find_child_by_name(self._nodes_at_level[layer_info.depth - 1][-1], name):
                        self._nodes_at_level[layer_info.depth][-1].name =\
                            f'{name}_{postfixes_at_level[layer_info.depth][name]}'

                    self._nodes_at_level[layer_info.depth][-1].parent =\
                                self._nodes_at_level[layer_info.depth - 1][-1]

                postfixes_at_level[layer_info.depth][name] += 1

        self._initialized = True


    def show_model_tree(self,
                            node_name_or_path: str = '',
                            max_depth: int = 0,
                            all_attrs: bool = False,
                            attr_list: Iterable[str] = [],
                            attr_omit_null: bool = False,
                            attr_bracket: List[str] = ['[', ']'],
                            style: str = 'const',
                            custom_style: Iterable[str] = [],) -> None:

        if not self._initialized:
            raise RuntimeError('TorchinfoWriter is not initialized. '
                                'Initialization is performed by calling '
                                'construct_model_tree(), which must be '
                                'done at least once before accessing '
                                'the summary data')

        self._nodes_at_level[0][0].show(
                            node_name_or_path=node_name_or_path,
                            max_depth=max_depth,
                            all_attrs=all_attrs,
                            attr_list=attr_list,
                            attr_bracket=attr_bracket,
                            style=style,
                            custom_style=custom_style)


    def get_dataframe(self,
                        strip_leading_dot: bool = False) -> pd.DataFrame:

        if not self._initialized:
            raise RuntimeError('TorchinfoWriter is not initialized. '
                                'Initialization is performed by calling '
                                'construct_model_tree(), which must be '
                                'done at least once before accessing '
                                'the summary data')

        model_summary_pd = tree_to_dataframe(self._nodes_at_level[0][0],
                                                            path_col='Path',
                                                            name_col='Name',
                                                            all_attrs=True)

        if strip_leading_dot:
            model_summary_pd['Path'] =\
                [path.lstrip('.') for path in model_summary_pd['Path']]

        return model_summary_pd.set_index('Path')


    def get_dot(self,
                    directed: bool = True,
                    rankdir: str = "TB",
                    bg_color: str = "",
                    node_color: str = "",
                    node_shape: str = "",
                    edge_color: str = "",
                    node_attr: str = "",
                    edge_attr: str = "") -> pydot.Dot:

        if not self._initialized:
            raise RuntimeError('TorchinfoWriter is not initialized. '
                                'Initialization is performed by calling '
                                'construct_model_tree(), which must be '
                                'done at least once before accessing '
                                'the summary data')

        return tree_to_dot(self._nodes_at_level[0][0],
                                    directed=directed,
                                    rankdir=rankdir,
                                    bg_colour=bg_color,
                                    node_colour=node_color,
                                    node_shape=node_shape,
                                    edge_colour=edge_color,
                                    node_attr=node_attr,
                                    edge_attr=edge_attr)


    def _label_to_row_entries(self, label: str) -> str:
        parts = label.split(' (')
        type_ = parts[0]
        name = parts[1][:-1]
        return name, type_
