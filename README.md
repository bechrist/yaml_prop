# yaml_prop
Extended `PyYAML` loader and dumper for properties with physical units used in scientific computing.

This utility extends PyYAML with the following YAML tags:
- `!array`: Takes a `yaml.SequenceNode` and converts it into the array or tensor object of your preference with [`array_yaml_constructor()`](/yaml_prop/math.py). The default array constructor is `numpy.array`.
- `!numexpr`: Leverages the [`numexpr` library](https://github.com/pydata/numexpr) to evaluate `numpy` expressions from a YAML mapping node with an `expr` key and an optional `alias` key for resolving aliases within the expression.
- `!lambda`: Creates a [`Lambda`](/yaml_prop/math.py) expression via `numexpr` from a YAML mapping node with `args`, `expr`, and `alias` keys.

These basic utilities are then used to define physical properties with physical unit conversions handled by [`pint`](https://github.com/hgrecco/pint)using the [`UNITS`](/yaml_prop/units.py) registry:
- `!constant`: Creates a constant physical property represented by a [`ConstantProperty`](/yaml_prop/properties.py).
- `!table`: Creates a lookup table physical property based on an underlying [`scipy.interpolate.RegularGridInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html). See [`TableProperty`](/yaml_prop/properties.py).
- `!function`: Wraps around a `Lambda` object and performs unit conversions. Refer to [`FunctionProperty`](/yaml_prop/properties.py).

The basic utilities and physical properties are included in [`PropertyLoader`](/yaml_prop/main.py) and [`PropertyDumper`](/yaml_prop/main.py).

---

Copyright (C) 2024-, The University of Texas at Austin

All Rights reserved.
See file COPYRIGHT for details.

This file is part of the yaml_prop package. For more information see
https://github.com/bechrist/yaml_prop

yaml_prop is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License (as published by the Free Software Foundation) version 3.0 dated June 2007.
