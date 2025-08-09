"""Physical units

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
__all__ = ['UNITS']

from collections.abc import Collection
import pint


# %%
class Units:
    def __init__(self):
        self.registry = pint.UnitRegistry(system='mks')
        self.registry.autoconvert_offset_to_baseunit = True
        self.registry.default_preferred_units = [
            self.registry.m,        # distance    
            self.registry.kg,       # mass          
            self.registry.s,        # time         
            self.registry.K,        # temperature
            self.registry.N,        # force
            self.registry.Pa,       # pressure
            self.registry.J,        # energy
            self.registry.W,        # power 
            self.registry.C,        # charge
            self.registry.V,        # voltage
            self.registry.Ω,        # resistance 
            self.registry.S,        # conductivity
            self.registry.F,        # capacitance
            self.registry.Wb,       # magnetic flux
            self.registry.H,        # inductance
            self.registry.T,        # magnetic field
            self.registry.Unit('kg/m^3'),   # density
            self.registry.Unit('mm/mm/K'),  # coefficient of thermal expansion
            self.registry.Unit('J/kg/K'),   # specific heat capacity
            self.registry.Unit('W/m/K'),    # thermal conductivity
            self.registry.Unit('Ω m')]      # electrical resistivity
        self.preferred = {qp.dimensionality: qp for qp in self.registry.default_preferred_units}

    def to(self, value: float | Collection[float], old_unit: str, new_unit: str) \
            -> float | Collection[float]:
        """Alias for :code:`Units.base()` method.
        
        :param value: Values in specified units
        :type value: float | Collection[float]
        
        :param old_unit: String representing old unit
        :type old_unit: str
        
        :param new_unit: String representing new unit
        :type new_unit: str

        :return: Values in new unit 
        :type return: float | Collection[float]
        """
        try:
            q = self.registry.Quantity(value, old_unit)
        except:
            q = value * self.registry.Quantity(old_unit)    
        q.ito(new_unit)
        return q.magnitude 
    
    def base(self, value: float | Collection[float], unit: str) \
            -> tuple[float | Collection[float], str]:
        """Converts value in specified units to value in base units.
        
        :param value: Values in specified units
        :type value: float | Collection[float]
        
        :param unit: String representing unit
        :type unit: str
        
        :return: Values in base units and corresponding base units
        :type return: tuple[float | Collection[float], str]
        """
        try:
            q = self.registry.Quantity(value, unit)
        except:
            q = value * self.registry.Quantity(unit)    
        q.ito_base_units()
        return q.magnitude, q.units
    
    def display(self, value: float | Collection[float], unit: str) \
            -> tuple[float | Collection[float], str]:
        """Converts value in specified units to value in compact preferred units.
        
        WARNING: Conversion into preferred units is slow!

        :param value: Values in specified units
        :type value: float | Collection[float]
        
        :param unit: String representing unit
        :type unit: str
        
        :return: Values in preferred compact units and corresponding units
        :type return: tuple[float | Collection[float], str]
        """
        try:
            q = self.registry.Quantity(value, unit)
        except:
            q = value * self.registry.Quantity(unit)
        
        if q.dimensionality in self.preferred:
            q.ito(self.preferred[q.dimensionality])

        # min_compact_unit = min(q).to_compact().units
        # max_compact_unit = max(q).to_compact().units
        # if min_compact_unit == max_compact_unit:
        #     q.ito(min_compact_unit)

        return q.magnitude, q.units
    

UNITS = Units()