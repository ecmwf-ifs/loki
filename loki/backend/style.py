# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pydantic.dataclasses import dataclass

__all__ = ['DefaultStyle', 'FortranStyle']

@dataclass
class DefaultStyle:
    """
    Default style class that defines the formatting of generated code.
    """
    linewidth: int = 90

    indent_default: int = 1
    indent_char: str = '  '


class FortranStyle(DefaultStyle):
    """
    Style class that defines the output code style for a Fortran backend.
    """
    linewidth: int = 132
