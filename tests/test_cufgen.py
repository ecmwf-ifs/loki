import pytest

from conftest import available_frontends
from loki import Module, FindNodes, Transformer
from loki import ir
from loki.expression import symbols as sym


@pytest.mark.parametrize('frontend', available_frontends())
def test_cufgen(frontend):
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

    module = Module.from_source(fcode, frontend=frontend)

    driver = None
    kernels = []
    device_subroutines = []
    for routine in module.routines:
        if "driver" in routine.name:
            driver = routine
        elif "kernel" in routine.name:
            kernels.append(routine)
        elif "device" in routine.name:
            device_subroutines.append(routine)

    assert driver
    assert module.to_fortran(cuf=True) == module.to_fortran()

    decl_map = {}
    for decl in FindNodes(ir.VariableDeclaration).visit(driver.spec):
        if "device" in decl.symbols[0].name:
            decl_map[decl] = decl.clone(symbols=(decl.symbols[0].clone(type=decl.symbols[0].type.clone(device=True)),))
        if "managed" in decl.symbols[0].name:
            decl_map[decl] = decl.clone(symbols=(decl.symbols[0].clone(type=decl.symbols[0].type.clone(managed=True)),))
        if "constant" in decl.symbols[0].name:
            decl_map[decl] = decl.clone(
                symbols=(decl.symbols[0].clone(type=decl.symbols[0].type.clone(constant=True)),))
        if "shared" in decl.symbols[0].name:
            decl_map[decl] = decl.clone(symbols=(decl.symbols[0].clone(type=decl.symbols[0].type.clone(shared=True)),))
        if "pinned" in decl.symbols[0].name:
            decl_map[decl] = decl.clone(symbols=(decl.symbols[0].clone(type=decl.symbols[0].type.clone(pinned=True)),))
        if "texture" in decl.symbols[0].name:
            decl_map[decl] = decl.clone(symbols=(decl.symbols[0].clone(type=decl.symbols[0].type.clone(texture=True)),))
    driver.spec = Transformer(decl_map).visit(driver.spec)

    call_map = {}
    for call in FindNodes(ir.CallStatement).visit(driver.body):
        if "kernel" in str(call.name):
            with pytest.raises(Exception):
                _ = call.clone(chevron=(sym.IntLiteral(1), sym.IntLiteral(1), sym.IntLiteral(1), sym.IntLiteral(1),
                                        sym.IntLiteral(1)))
            with pytest.raises(Exception):
                _ = call.clone(chevron=(1, 1))
            with pytest.raises(Exception):
                _ = call.clone(chevron=2)

            call_map[call] = call.clone(chevron=(sym.IntLiteral(1), sym.IntLiteral(1),
                                                 sym.IntLiteral(1), sym.IntLiteral(1)))

    driver.body = Transformer(call_map).visit(driver.body)

    for kernel in kernels:
        kernel.prefix = ("ATTRIBUTES(GLOBAL)",)

    for device_subroutine in device_subroutines:
        device_subroutine.prefix = ("ATTRIBUTES(DEVICE)",)

    cuf_str = module.to_fortran(cuf=True)

    assert "INTEGER, DEVICE" in cuf_str
    assert "INTEGER, MANAGED" in cuf_str
    assert "INTEGER, CONSTANT" in cuf_str
    assert "INTEGER, SHARED" in cuf_str
    assert "INTEGER, PINNED" in cuf_str
    assert "INTEGER, TEXTURE" in cuf_str

    assert "<<<" in cuf_str and ">>>" in cuf_str

    assert "ATTRIBUTES(GLOBAL) SUBROUTINE kernel" in cuf_str
    assert "ATTRIBUTES(DEVICE) SUBROUTINE device" in cuf_str
