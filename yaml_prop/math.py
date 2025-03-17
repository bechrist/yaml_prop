"""Math expression evaluation

Copyright (C) 2024-, The University of Texas at Austin

All Rights reserved.
See file COPYRIGHT for details.

This file is part of the yaml_prop package. For more information see
https://github.com/bechrist/yaml_prop

yaml_prop is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License (as published by the Free
Software Foundation) version 3.0 dated June 2007.
"""
__authors__ = ['Blake Christierson, UT Austin <bechristierson@utexas.edu>']
__all__ = ['array_yaml_constructor', 'numpy_array_yaml_representer',
           'numexpr_yaml_constructor', 'Lambda']

import typing as typ
from collections.abc import Callable, Mapping, Sequence
import numexpr as ne
import numpy as np
from typing import Any
import yaml

from .common import YAMLObject


# %%
_Array = np.ndarray | Any 
_ArrayConstructor = Callable[[list], _Array]


def array_yaml_constructor(loader: yaml.Loader, node: yaml.SequenceNode,
                           constructor: _ArrayConstructor = np.array) -> _Array:
    """Constructs an array from YAML sequence(s)

    :param loader: YAML loader
    :type loader: yaml.Loader

    :param node: YAML (nested) sequence node
    :type node: yaml.SequenceNode

    :param constructor: Array constructor, defaults to :code:`numpy.array`
    :type constructor: _ArrayConstructor
    
    :return: Array
    :rtype: _Array
    """
    return constructor(loader.construct_sequence(node, deep=True))


def numpy_array_yaml_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.SequenceNode:
    """Represents :code:`numpy.ndarray` by YAML sequence(s)

    :param dumper: YAML dumper
    :type dumper: yaml.Dumper

    :param array: :code:`numpy` Array
    :type array: numpy.ndarray

    :return: YAML (nested) sequence node
    :rtype: yaml.SequenceNode
    """
    return dumper.represent_list(array.to_list())


# %%
def numexpr_yaml_constructor(loader: yaml.Loader, node: yaml.MappingNode) -> np.ndarray:
    """Leverages :code:`numexpr` to compute :code:`numpy` expressions from YAML.

    :param loader: YAML loader
    :type loader: yaml.Loader

    :param node: YAML mapping node
    :type node: yaml.MappingNode
    
    :return: Numpy array
    :rtype: numpy.ndarray
    """
    mapping = {'expr': '', 'alias': {}}
    mapping.update(loader.construct_mapping(node, deep=True))
    return ne.evaluate(mapping['expr'], local_dict=mapping['alias'], global_dict={}) 


# %%
class Lambda(YAMLObject):
    """Lambda expression used for evaluating functional relationships

    :param args: Lambda arguments
    :type args: typing.Sequence[str]

    :param expr: Lambda expression (must be :code:`numexpr` compatible)
    :type expr: str

    :param alias: Expression aliases to be resolved, defaults to :code:`dict()`
    :type alias: typing.Mapping[str, typing.Any], optional
    """
    _yaml_tag = u'!lambda'
    _yaml_attrs = ('args', 'expr', 'alias')

    def __init__(self, args: Sequence[str], expr: str, 
                 alias: Mapping[str, Any] = None):
        """Initializes :code:`Lambda`, see class docstring"""
        self.args = tuple(args)
        self.expr = expr
        self.alias = dict() if alias is None else alias

        self.local_dict = lambda *args: self.alias | {a: arg for a, arg in zip(self.args, args)}

    def __call__(self, *args) -> np.ndarray:
        """Evaluates :code:`Lambda`
        
        :param *args: Arguments in order of :code:`self.args`
        
        :returns: Evaluation
        :rtype: numpy.ndarray
        """
        return ne.evaluate(self.expr, local_dict=self.local_dict(args), global_dict={})
