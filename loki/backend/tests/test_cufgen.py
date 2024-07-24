# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import ValidationError

from loki import Module
from loki.ir import nodes as ir, FindNodes, Transformer
from loki.expression import symbols as sym
from loki.frontend import available_frontends


@pytest.mark.parametrize('frontend', available_frontends())
def test_cufgen(frontend, tmp_path):
    """
    A simple test routine to test the Cuda Fortran (CUF) backend
    """

    fcode = """
module transformation_module_cufgen
  implicit none
  integer, parameter :: len = 10
contains

  subroutine driver(a, b, c)
    integer, intent(inout) :: a
    integer, intent(inout) :: b(len)
    integer, intent(inout) :: c(a, len)
    integer :: var_device
    integer :: var_managed
    integer :: var_constant
    integer :: var_shared
    integer :: var_pinned
    integer :: var_texture
    call kernel(a, b)
  end subroutine driver

  subroutine kernel(a, b)
    integer, intent(inout) :: a
    integer, intent(inout) :: b(len)
    real :: x(a)
    real :: k2_tmp(a, a)
    call device1(x, k2_tmp)
  end subroutine kernel

  subroutine device(x, y)
    real, intent(inout) :: x(len)
    real, intent(inout) :: y(len, len)
  end subroutine device

end module transformation_module_cufgen
"""

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    driver = module['driver']
    kernel = module['kernel']
    device_subroutine = module['device']

    assert driver
    assert module.to_fortran(cuf=True) == module.to_fortran()

    for var in driver.variables:
        if "device" in var.name:
            var.type = var.type.clone(device=True)
        if "managed" in var.name:
            var.type = var.type.clone(managed=True)
        if "constant" in var.name:
            var.type = var.type.clone(constant=True)
        if "shared" in var.name:
            var.type = var.type.clone(shared=True)
        if "pinned" in var.name:
            var.type = var.type.clone(pinned=True)
        if "texture" in var.name:
            var.type = var.type.clone(texture=True)

    call_map = {}
    for call in FindNodes(ir.CallStatement).visit(driver.body):
        if "kernel" in str(call.name):
            with pytest.raises(ValidationError):
                _ = call.clone(chevron=(sym.IntLiteral(1), sym.IntLiteral(1), sym.IntLiteral(1), sym.IntLiteral(1),
                                        sym.IntLiteral(1)))
            with pytest.raises(ValidationError):
                _ = call.clone(chevron=(1, 1))
            with pytest.raises(ValidationError):
                _ = call.clone(chevron=2)

            call_map[call] = call.clone(chevron=(sym.IntLiteral(1), sym.IntLiteral(1),
                                                 sym.IntLiteral(1), sym.IntLiteral(1)))

    driver.body = Transformer(call_map).visit(driver.body)

    kernel.prefix = ("ATTRIBUTES(GLOBAL)",)
    device_subroutine.prefix = ("ATTRIBUTES(DEVICE)",)

    cuf_driver_str = driver.to_fortran(cuf=True)
    cuf_kernel_str = kernel.to_fortran(cuf=True)
    cuf_device_str = device_subroutine.to_fortran(cuf=True)

    assert "INTEGER, DEVICE" in cuf_driver_str
    assert "INTEGER, MANAGED" in cuf_driver_str
    assert "INTEGER, CONSTANT" in cuf_driver_str
    assert "INTEGER, SHARED" in cuf_driver_str
    assert "INTEGER, PINNED" in cuf_driver_str
    assert "INTEGER, TEXTURE" in cuf_driver_str

    assert "<<<" in cuf_driver_str and ">>>" in cuf_driver_str

    assert "ATTRIBUTES(GLOBAL) SUBROUTINE kernel" in cuf_kernel_str
    assert "ATTRIBUTES(DEVICE) SUBROUTINE device" in cuf_device_str
