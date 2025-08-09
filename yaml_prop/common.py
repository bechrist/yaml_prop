"""YAML object class

Copyright (C) 2024-, The University of Texas at Austin

All Rights reserved.
See file COPYRIGHT for details.

This file is part of the yaml_prop package. For more information see
https://github.com/bechrist/yaml_prop

yaml_prop is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License (as published by the Free
Software Foundation) version 3.0 dated June 2007.
"""
from __future__ import annotations

__authors__ = ['Blake Christierson, UT Austin <bechristierson@utexas.edu>']
__all__ = ['YAMLObject']

import yaml

# %%
class YAMLObject:
    _yaml_tag: str = None
    _yaml_attrs: tuple[str] = tuple()

    @classmethod
    def yaml_constructor(cls, loader: yaml.Loader, node: yaml.MappingNode) -> YAMLObject:
        """Constructs :code:`YAMLObject` from YAML node

        :param loader: YAML loader
        :type loader: yaml.Loader

        :param node: YAML node
        :type node: yaml.MappingNode

        :return: YAML object
        :rtype: YAMLObject
        """
        return cls(**loader.construct_mapping(node, deep=True))
    
    @classmethod
    def yaml_representer(cls, dumper: yaml.Dumper, obj: YAMLObject) -> yaml.MappingNode:
        """Represents :code:`YAMLObject` in a YAML node

        :param dumper: YAML dumper
        :type dumper: yaml.Dumper

        :param obj: YAML object
        :type obj: YAMLObject

        :return: YAML node representation
        :rtype: yaml.Node
        """
        attrs = {attr: getattr(obj, attr) for attr in cls._yaml_attrs}
        return dumper.represent_mapping(cls._yaml_tag, attrs)