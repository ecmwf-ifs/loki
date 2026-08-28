# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pydantic.dataclasses import dataclass

__all__ = ['DefaultStyle', 'FortranStyle', 'IFSFortranStyle']

@dataclass
class DefaultStyle:
    """
    Default style class that defines the formatting of generated code.
    """
    linewidth: int = 90

    indent_default: int = 2
    indent_char: str = ' '


@dataclass
class FortranStyle(DefaultStyle):
    """
    Style class that defines the output code style for a Fortran backend.
    """
    linewidth: int = 132

    associate_indent: int = 2

    conditional_indent: int = 2
    conditional_end_space: bool = True

    loop_indent: int = 2
    loop_end_space: bool = True

    procedure_spec_indent: int = 2
    procedure_body_indent: int = 2
    procedure_contains_indent: int = 2
    procedure_end_named: bool = True

    module_spec_indent: int = 2
    module_contains_indent: int = 2
    module_end_named: bool = True


@dataclass
class IFSFortranStyle(FortranStyle):
    """
    Style class that defines the output code style for a Fortran backend.
    """
    linewidth: int = 132

    associate_indent: int = 0

    conditional_indent: int = 2
    conditional_end_space: bool = False

    loop_indent: int = 2
    loop_end_space: bool = False

    procedure_spec_indent: int = 0
    procedure_body_indent: int = 0
    procedure_contains_indent: int = 2
    procedure_end_named: bool = True

    module_spec_indent: int = 0
    module_contains_indent: int = 2
    module_end_named: bool = True
