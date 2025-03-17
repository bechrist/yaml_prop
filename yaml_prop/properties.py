"""Property YAML object definitions

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
__all__ = ['ConstantProperty', 'TableProperty', 'FunctionProperty', '_Property']

from collections.abc import Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi

from .common import YAMLObject
from .units import UNITS


# %%
class ConstantProperty(YAMLObject):
    """Constant physical property

    :param name: Property name
    :type name: str
    
    :param unit: Property unit
    :type unit: str
    
    :param symbol: Property symbol
    :type symbol: str

    :param value: Property value
    :type value: float | numpy.ndarray
    """
    _yaml_tag = u'!constant'
    _yaml_attrs = ('name', 'unit', 'symbol', 'value')

    def __init__(self, name: str, unit: str, symbol: str, value: float | np.ndarray):
        """Initializes :code:`ConstantProperty`, see class docstring"""
        self.name = name
        self.symbol = symbol
        self.value, self.unit = UNITS.base(value, unit)

    def __call__(self, *_, **__) -> float | np.ndarray:
        """Returns property value
        
        :return: Property value
        :rtype: float | numpy.ndarray
        """
        return self.value
    
    def plot(self, *_, **__):
        """Plotting method"""
        pass

class TableProperty(YAMLObject):
    """Table property supporting :math:`n`-dimensional gridded interpolation.

    :param name: Property name
    :type name: str

    :param arguments: Property argument names
    :type arguments: Sequence[str]

    :param units: Argument and property units
    :type units: Sequence[str]

    :param symbols: Argument and property symbols
    :type symbols: Sequence[str]

    :param defaults: Default argument values
    :type defaults: Sequence[float | numpy.ndarray]
    
    :param values: Gridded interpolant values
    :type values: Sequence[numpy.ndarray]

    :param method: Interpolation method, defaults to :code:`'linear'`
    :type method: str, optional
    """
    _yaml_tag = u'!table'
    _yaml_attrs = ('name', 'arguments', 'units', 'symbols', 'defaults', 'values', 'method')

    def __init__(self, 
            name: str,
            arguments: Sequence[str], 
            units: Sequence[str], 
            symbols: Sequence[str], 
            defaults: Sequence[float | np.ndarray], 
            values: Sequence[np.ndarray], 
            method: str = 'linear'):
        """Initializes :code:`TableProperty`, see class docstring"""
        self.name = name
        self.arguments = tuple(arguments)
        self._arguments = tuple(arg.lower() for arg in self.arguments)
        self.symbols = tuple(symbols)
        self.method = method 

        self.values, self.units = [], []
        for v, u in zip(values, units):
            value, unit = UNITS.base(v, u)
            self.values.append(value)
            self.units.append(unit)

        self.defaults = []
        for d, u_old, u in zip(defaults, units, self.units):
            self.defaults.append(UNITS.to(d, u_old, u))

        self.units = tuple(self.units)
        self.defaults = np.array(self.defaults)
        self.values = tuple(np.array(v) for v in self.values)

        self.min = np.array([v.min() for v in self.values[:-1]])
        self.max = np.array([v.max() for v in self.values[:-1]])
        self.interp = spi.RegularGridInterpolator(self.values[:-1], self.values[-1],
                                                  method=self.method, bounds_error=False)
    
    def __call__(self, *args, **kwargs) -> float | np.ndarray:
        """Interpolates table property value(s)

        :param args: Positional arguments
        :param kwargs: Keyword arguments

        :return: Interpolated table value(s)
        :rtype: float
        """
        x = _parse_prop_args(self, *args, **kwargs)
        for i, xi in enumerate(x.T):
            for j, xij in enumerate(xi):
                if xij < self.min[i]:
                    print(f'WARNING: Thesholding {j}-th {self.arguments[i]} to minimum: {xij} < {self.min[i]}')
                    x[i][j] = self.min[i]
                if self.max[i] < xij:
                    print(f'WARNING: Thesholding {j}-th {self.arguments[i]} to maximum: {xij} > {self.max[i]}')
                    x[i][j] = self.max[i]
        return self.interp(x)

    def plot(self, argument: str, units: tuple[str, ...] | None = None, **kwargs) \
            -> tuple[plt.PathCollection, tuple[str, ...]]: 
        """Generates scatter plot and returns plot object and display units

        :param argument: Independent argument name
        :type argument: str

        :param units: Display units, defaults to property display units
        :type units: tuple[str, ...], optional
        
        :param kwargs: :code:`matplotlib.pyplot.scatter` kwargs

        :return: Scatter plot handle and display units
        :rtype: tuple[matplotlib.pyplot.PathCollection, tuple[str, ...]]
        """
        argument = argument.lower()
        idx = self._arguments.index(argument)
        xq = self.values[idx]
        x = {a: np.full(xq.shape, d) for a, d in zip(self._arguments, self.defaults)}
        x[argument] = xq
    
        y = self(**x)
        x, y, units = _convert_to_display_values(x, y, self, units)
        s = plt.scatter(x[argument], y, **kwargs)
        return s, units
    
        
class FunctionProperty(YAMLObject):
    """Functional physical property

    :param name: Property name
    :type name: str

    :param arguments: Property argument names
    :type arguments: Sequence[str]

    :param units: Argument and property units
    :type units: Sequence[str]

    :param symbols: Argument and property symbols
    :type symbols: Sequence[str]

    :param defaults: Default argument values
    :type defaults: Sequence[float | numpy.ndarray]

    :param bounds: Argument bounds
    :type bounds: Sequence[Sequence[float]]

    :param expression: Function expression
    :type expression: Callable
    """
    _yaml_tag = u"!function"
    _yaml_attrs = ('name', 'arguments', 'units', 'symbols', 'defaults', 'bounds', 'expression')

    def __init__(self, 
            name: str,
            arguments: Sequence[str], 
            units: Sequence[str], 
            symbols: Sequence[str], 
            defaults: Sequence[float],
            bounds: Sequence[Sequence[float]], 
            expression: Callable):
        """Initializes :code:`FunctionProperty`, see class docstring"""
        self.name = name
        self.arguments = tuple(arguments)
        self._arguments = tuple(arg.lower() for arg in self.arguments)
        self.symbols = tuple(symbols)
        self.expression = expression # TODO: expression unit conversions
    
        self._old_units = units
        self.defaults, self.units = [], []
        for d, u in zip(defaults, units):
            default, unit = UNITS.base(d, u)
            self.defaults.append(default)
            self.units.append(unit)

        _, unit = UNITS.base(0, units[-1])
        self.units.append(unit)

        self.bounds = []
        for b, u_old, u in zip(bounds, units, self.units):
            self.bounds.append(UNITS.to(b, u_old, u))

        self.units = tuple(self.units)
        self.defaults = np.array(self.defaults)
        self.bounds = np.array(self.bounds)
    
    def __call__(self, *args, **kwargs) -> float | np.ndarray:
        """Evaluates expression with physical unit conversions

        :param args: Positional arguments
        :param kwargs: Keyword arguments

        :return: Evaluated expression value(s)
        :rtype: float | numpy.ndarray
        """
        x = _parse_prop_args(self, *args, **kwargs)
        for i, xi in enumerate(x.T):
            x[i:] = UNITS.to(xi, self.units[i], self._old_units[i]).reshape(-1,1)
            if any(xi < self.bounds[i][0]) or any(self.bounds[i][1] < xi):
                raise ValueError("Evaluation point is outside of bounds")
        return UNITS.to(self.expression(x), self._old_units[-1], self.units[-1])[0]
    
    def plot(self, argument: str, units: tuple[str, ...] | None = None, **kwargs) \
            -> tuple[list[plt.LineCollection], tuple[str, ...]]:
        """Generates plot and returns plot object and display units

        :param argument: Independent argument name
        :type argument: str

        :param units: Display units, defaults to property display units
        :type units: tuple[str, ...], optional
        
        :param kwargs: :code:`matplotlib.pyplot.plot` kwargs

        :return: Plot handles and display units
        :rtype: tuple[list[matplotlib.pyplot.LineCollection], tuple[str, ...]]
        """
        argument = argument.lower()
        idx = self._arguments.index(argument)
        xq = np.linspace(*self.bounds[idx], 1000)
        x = {a: np.full(xq.shape, d) for a, d in zip(self._arguments, self.defaults)}
        x[argument] = xq
    
        y = self(**x)
        x, y, units = _convert_to_display_values(x, y, self, units)
        l = plt.plot(x[argument], y, **kwargs)
        return l, units
    

# %%
_Property = ConstantProperty | TableProperty | FunctionProperty


def _parse_prop_args(prop: TableProperty | FunctionProperty, *args, **kwargs) -> np.ndarray:
    """Parses property evaluation arguments

    :param prop: Property
    :type prop: TableProperty | FunctionProperty

    :param args: Positional arguments
    :param kwargs: Keyword arguments

    :return: Argument array
    :rtype: numpy.ndarray
    """
    arguments, i = {}, 0
    for i, k in enumerate(prop._arguments):
        if i < len(args):
            if k in kwargs:
                raise ValueError(f"'{prop.arguments[i]}' values in args and kwargs")
            arguments[k] = args
        elif k in kwargs:
            arguments[k] = kwargs[k]
        else:
            arguments[k] = prop.defaults[i]

    return np.array([arguments[k] for k in prop._arguments]).T


def _convert_to_display_values(x: np.ndarray, y: np.ndarray,
                               prop: TableProperty | FunctionProperty,
                               units: tuple[str, ...] | None = None) \
                               -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Helper function to convert plotting data to display units

    :param x: Independent variable values
    :type x: numpy.ndarray

    :param y: Dependent variable values
    :type y: numpy.ndarray

    :param prop: Property
    :type prop: TableProperty | FunctionProperty
    
    :param units: Display units, defaults to :code:`prop.units`
    :type units: tuple[str, ...], optional

    :return: Display unit variable values and units
    :rtype: tuple[numpy.ndarray, numpy.ndarray, tuple[str, ...]]
    """
    if units is None:
        units = [None]*len(prop.units)
        for k, xk in x.items():
            if k not in units: 
                i = prop._arguments.index(k)
                x[k], u = UNITS.display(xk, prop.units[i])
                units[i] = u

        y, u = UNITS.display(y, prop.units[-1])
        units[-1] = u
    else:
        for k, xk in x.items():
            i = prop._arguments.index(k)
            x[k] = UNITS.to(xk, prop.units[i], units[i])
        
        y = UNITS.to(y, prop.units[-1], units[-1])

    return x, y, tuple(units)
