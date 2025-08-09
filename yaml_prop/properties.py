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

from collections.abc import Callable, Sequence, Mapping
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from numpy.typing import ArrayLike

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

    def __init__(self, name: str, unit: str, symbol: str, value: float | np.ndarray, *_, **__):
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
    """Table property supporting :math:`n`-dimensional linear interpolation.

    :param name: Property name
    :type name: str

    :param arguments: Property argument names
    :type arguments: Sequence[str]

    :param units: Argument and property units
    :type units: Sequence[str]

    :param symbols: Argument and property symbols
    :type symbols: Sequence[str]

    :param defaults: Default argument values
    :type defaults: Sequence[float]
    
    :param values: Gridded interpolant values
    :type values: Sequence[numpy.ndarray]

    :param order: Value argument listing order , defaults to :code:`None`
    :type order: Sequence[int], optional
    """
    _yaml_tag = u'!table'
    _yaml_attrs = ('name', 'arguments', 'units', 'symbols', 'defaults', 'values', 'order')

    def __init__(self, 
        name: str,
        arguments: Sequence[str], 
        units: Sequence[str], 
        symbols: Sequence[str], 
        defaults: Sequence[float], 
        values: Sequence[Sequence[float | Sequence]] | Mapping[float, Sequence[float | Sequence] | Mapping], 
        order: Sequence[int] = None,
        *_, **__,
    ) -> None:
        """Initializes :code:`TableProperty`, see class docstring"""
        self.name = name
        self.arguments = tuple(arguments)
        self.symbols = tuple(symbols)

        self._arguments = tuple(arg.lower() for arg in self.arguments)
        self.ndim = len(self.arguments)

        self.order = range(self.ndim) if order is None else order
        if len(self.order) != self.ndim:
            raise ValueError('Length of value order does not correspond to `ndim`')
        
        self.order = tuple(self.order)
        self.iorder = tuple(np.argsort(self.order))

        self.gridded = False
        if isinstance(values, Sequence):
            self.gridded = True
            self._setup_gridded(values, units)
        elif isinstance(values, Mapping):
            self._setup_scattered(values, units)
        else:
            raise TypeError(f"Invalid type for `values`: {type(values)}")

        self.defaults = []
        for d, u_, u in zip(defaults, units, self.units):
            self.defaults.append(UNITS.to(d, u_, u))
        self.defaults = np.array(self.defaults)

    def _setup_gridded(self, values: Sequence[ArrayLike], units: Sequence[str]) -> None:
        """Performs unit conversions and builds interpolant for gridded data

        :param values: Gridded data
        :type values: Sequence[ArrayLike]

        :param units: Physical units
        :type units: Sequence[str]
        """
        # Convert values
        self.values = []
        self.units = [None for _ in range(self.ndim)]
        for j, v in zip(self.order, values[:-1]):
            value, self.units[j] = UNITS.base(v, units[j])
            self.values.append(value)
        
        value, unit = UNITS.base(values[-1], units[-1])
        self.values.append(value)
        self.units.append(unit)

        self.values = tuple(np.array(v) for v in self.values)
        self.units = tuple(self.units)
        
        # Setup interpolant
        self.min = np.array([v.min() for v in self.values[:-1]])
        self.max = np.array([v.max() for v in self.values[:-1]])

        self.interp = RegularGridInterpolator(self.values[:-1], self.values[-1])
    
    def _setup_scattered(self,
        values: Mapping[float | Sequence[float], float | Sequence[float] | Mapping],
        units: Sequence[str],
    ) -> None:
        """Performs unit conversions and builds interpolant for scattered data

        :param values: Scattered data
        :type values: Mapping[float | Sequence[float], float | Sequence[float] | Mapping]

        :param units: Physical units
        :type units: Sequence[str]
        """
        # Parse values
        self.values = [[],[]]
        def __parse_scattered(value, point: Sequence[float] = ()) -> None:
            """Flattens nested mappings into `self.values`"""
            if isinstance(value, Mapping):
                for k, v in value.items():
                    __parse_scattered(v, point + ((*k,) if isinstance(k, Sequence) else (k,)))
            else:
                self.values[0].append(point)
                self.values[1].append(value)
        __parse_scattered(values)

        self.values = [np.array(v) for v in self.values]

        # Convert values
        self.units = [None for _ in range(self.ndim)]
        for i, (j, v) in enumerate(zip(self.order, self.values[0].T)):
            self.values[0][:,i], self.units[j] = UNITS.base(v, units[j])

        self.values[1], unit = UNITS.base(self.values[1], units[-1])
        self.units.append(unit)

        self.values = tuple(self.values)
        self.units = tuple(self.units)

        # Setup interpolant
        self.min = np.array([v.min() for v in self.values[0].T])
        self.max = np.array([v.max() for v in self.values[0].T])

        self.interp = LinearNDInterpolator(self.values[0], self.values[1])

    def __call__(self, *args, threshold: bool = True, **kwargs) -> float | np.ndarray:
        """Interpolates table property value(s)

        :param args: Positional arguments

        :param threshold: Thresholds arguments to the support of table values, defaults to :code:`True`
        :type threshold: bool, optional

        :param kwargs: Keyword arguments

        :return: Interpolated table value(s)
        :rtype: float
        """
        x = _parse_prop_args(self, *args, **kwargs)

        if threshold:
            for i, xi in enumerate(x.T):
                for j, xij in enumerate(xi):
                    if xij < self.min[i]:
                        print(f'Thesholding {j}-th {self.arguments[i]} to minimum: {xij} < {self.min[i]}')
                        x[i][j] = self.min[i]
                    if self.max[i] < xij:
                        print(f'Thesholding {j}-th {self.arguments[i]} to maximum: {xij} > {self.max[i]}')
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
        if not self.gridded:
            return # TODO: rehash plotting
    
        argument = argument.lower()
        idx = self._arguments.index(argument)
        xq = self.values[idx]
        x = {a: np.full(xq.shape, d) for a, d in zip(self._arguments, self.defaults)}
        x[argument] = xq
    
        y = self(**x)
        x, y, units = _convert_to_display_values(x, y, self, units)
        s = plt.scatter(x[argument], y, **kwargs)
        return s, (units[idx], units[-1])
    
        
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
        expression: Callable,
        *_, **__,
    ) -> None:
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
        return l, (units[idx], units[-1])
    

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
                raise ValueError(f"`{prop.arguments[i]}` values in args and kwargs")
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