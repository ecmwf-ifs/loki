# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from loki.sourcefile import Sourcefile
from loki.expression import FindVariables, FindInlineCalls
from loki.ir import (
    CallStatement, Import,
)
from loki.visitors import FindNodes
from loki.transform import (
    extract_contained_procedures
)
from loki.subroutine import Subroutine

def test_extract_contained_procedures_basic_scalar():
    """
    Tests that a global scalar is correctly added as argument of `inner`.
    """
    fcode = """
        subroutine outer()
            implicit none
            integer :: x
            x = 42
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                z = x + y
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    assert routines[0].name == "inner"
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'x' in (var.name for var in inner.arguments)

    call = FindNodes(CallStatement).visit(outer.body)[0]
    assert 'x' in (var.name for var in call.arguments)

def test_extract_contained_procedures_contains_emptied():
    """
    Tests that the contains section does not contain any functions or subroutines after processing. 
    """
    fcode = """
        subroutine outer()
            implicit none
            integer :: x
            x = 42
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                z = x + y
            end subroutine inner
            function f() result(res)
                integer :: y
                integer :: z
                integer :: res
                y = 1
                z = y
                res = 2 * z
            end function f
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    outer = src.routines[0]
    extract_contained_procedures(outer)
    # NOTE: Functions in Loki are also typed as Subroutines.
    assert not any(isinstance(r, Subroutine) for r in outer.contains.body)

def test_extract_contained_procedures_basic_array():
    """
    Tests that a global array variable (and a scalar) is correctly added as argument of `inner`.
    """

    fcode = """
        subroutine outer()
            implicit none
            integer :: x
            real :: arr(3)
            arr = 71.0
            x = 42
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                z = x + y + arr(1)
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'x' in (var.name for var in inner.arguments)
    assert 'arr' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'arr' in (var.name for var in inner.arguments)

    call = FindNodes(CallStatement).visit(outer.body)[0]
    assert 'x' in (var.name for var in call.arguments)
    assert 'arr' in (var.name for var in call.arguments)

def test_extract_contained_procedures_basic_import():
    """
    Tests that a global imported binding is correctly introduced to the contained subroutine.
    """

    fcode = """
        subroutine outer()
            use constants, only: c1, c2
            implicit none
            integer :: x
            x = 42 + c1
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                z = x + y + c2
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    inner = routines[0]
    assert "c2" in inner.import_map.keys()
    assert not "c1" in inner.import_map.keys()
    assert not 'c2' in (var.name for var in inner.arguments)

def test_extract_contained_procedures_recursive_definition():
    """
    Tests that whenever a global in the contained subroutine depends on another
    global variable, both are introduced as arguments,
    even if there is no explicit reference to the latter.
    """
    fcode = """
        subroutine outer(klon)
            implicit none
            integer, intent(in) :: klon
            integer :: x(klon)
            x(klon - 1) = 42 
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                z = x(1) + y
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    outer = src.routines[0]
    inner = routines[0]
    assert 'x' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'x' in (var.name for var in inner.arguments)
    assert 'klon' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'klon' in (var.name for var in inner.arguments)

    call = FindNodes(CallStatement).visit(outer.body)[0]
    assert 'x' in (var.name for var in call.arguments)
    assert 'klon' in (var.name for var in call.arguments)

    # Test that intent of 'klon' is also 'in' inside inner (because intent is given in parent).
    klon = inner.variable_map['klon']
    assert klon.type.intent == "in"

def test_extract_contained_procedures_recursive_definition_import():
    """
    Tests that whenever globals in the contained subroutine depend on imported bindings, 
    the globals are introduced as arguments, and the imports are added to the contained subroutine. 
    """
    fcode = """
        subroutine outer()
            implicit none
            use parkind1, only: jprb, jpim 
            real(kind=jprb) :: x(3)
            integer(kind=jpim) :: ii(30)
            ii = 72
            x(1) = 42 
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                ii(4) = 2
                z = x(1) + y
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    outer = src.routines[0]
    inner = routines[0]
    assert 'x' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'x' in (var.name for var in inner.arguments)
    assert 'ii' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'ii' in (var.name for var in inner.arguments)
    call = FindNodes(CallStatement).visit(outer.body)[0]
    assert 'x' in (var.name for var in call.arguments)
    assert 'ii' in (var.name for var in call.arguments)

    imports = FindNodes(Import).visit(inner.spec)
    modules = set()
    symbols = set()
    for imp in imports:
        modules.add(imp.module)
        for sym in imp.symbols:
            symbols.add(sym)
    assert "parkind1" in modules
    assert len(modules) == 1
    assert "jprb" in symbols
    assert "jpim" in symbols
    assert len(symbols) == 2

def test_extract_contained_procedures_kind_resolution():
    """
    Tests that an unresolved kind parameter in inner scope is resolved from import in outer scope.
    """
    fcode = """
        subroutine outer()
            implicit none
            use parkind1, only: jpim
            call inner()
            contains
            subroutine inner()
                integer(kind = jpim) :: y
                integer :: z
                z = y
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    inner = routines[0]
    assert "jpim" in inner.import_map.keys()

def test_extract_contained_procedures_derived_type_resolution():
    """
    Tests that an unresolved derived type in inner scope is resolved from import in outer scope.
    """
    fcode = """
        subroutine outer()
            implicit none
            use stuff, only: mytype
            call inner()
            contains
            subroutine inner()
                type(mytype) :: y
                integer :: z
                z = y%a
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    inner = routines[0]
    assert "mytype" in inner.import_map.keys()

def test_extract_contained_procedures_derived_type_field():
    """
    Test that when a derived type field, i.e 'a%b' is a global in the scope of the contained subroutine,
    the derived type itself, that is, 'a', is introduced as an the argument in the transformation.
    """
    fcode = """
        subroutine outer()
            implicit none
            use types, only: my_type, your_type
            type(my_type) :: xtyp
            type(your_type) :: ytyp
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                xtyp%a = 40
                ytyp%val%b = 10.0
                z = y + ytyp%something_else
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    outer = src.routines[0]
    inner = routines[0]
    assert 'xtyp' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'xtyp' in (var.name for var in inner.arguments)
    assert 'ytyp' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'ytyp' in (var.name for var in inner.arguments)

    call = FindNodes(CallStatement).visit(outer.body)[0]
    assert 'xtyp' in (var.name for var in call.arguments)
    assert 'ytyp' in (var.name for var in call.arguments)

    imports = FindNodes(Import).visit(inner.spec)
    modules = set()
    symbols = set()
    for imp in imports:
        modules.add(imp.module)
        for sym in imp.symbols:
            symbols.add(sym)
    assert "types" in modules
    assert len(modules) == 1
    assert "my_type" in symbols
    assert "your_type" in symbols
    assert len(symbols) == 2

def test_extract_contained_procedures_intent():
    """
    This test is just to document the current behaviour: when a global is
    introduced as an argument to the lifted contained subroutine,
    its intent will be 'inout', unless the intent is specified in the parent subroutine.
    """
    fcode = """
        subroutine outer(v)
            implicit none
            integer, intent(in) :: v
            integer, intent(out) :: p
            integer :: x(3)
            x = 4
            call inner()
            p = 400
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                z = x(1) + v + y + p
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    outer = src.routines[0]
    inner = routines[0]
    assert inner.variable_map['v'].type.intent == "in"
    assert inner.variable_map['x'].type.intent == "inout"
    assert inner.variable_map['p'].type.intent == "out"

    # Also check that the intents don't change in the parent.
    assert outer.variable_map['v'].type.intent == "in"
    assert outer.variable_map['x'].type.intent is None
    assert outer.variable_map['p'].type.intent == "out"

def test_extract_contained_procedures_undefined_in_parent():
    """
    This test is just to document current behaviour:
    an exception is raised if a global inside the contained subroutine does not
    have a definition in the parent scope.
    """
    fcode = """
        subroutine outer()
            implicit none
            integer :: x
            x = 42
            call inner()
            contains
            subroutine inner()
                integer :: y
                y = 1
                z = x + y + g + f ! 'z', 'g', 'f' undefined in contained subroutine and parent.
            end subroutine inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    with pytest.raises(RuntimeError):
        extract_contained_procedures(src.routines[0])


def test_extract_contained_procedures_multiple_contained_procedures():
    """
    Basic test to check that multiple contained subroutines can also be handled.
    """
    fcode = """
        subroutine outer()
            implicit none
            integer :: x, gx
            x = 42
            gx = 10
            call inner1()
            call inner2()
            contains
            subroutine inner1()
                integer :: y
                integer :: z
                y = 1
                z = x + y
            end subroutine inner1
            subroutine inner2()
                integer :: gy
                integer :: gz
                gy = 1
                gz = gx + gy
            end subroutine inner2
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 2
    assert routines[0].name == "inner1"
    assert routines[1].name == "inner2"
    outer = src.routines[0]
    inner1 = routines[0]
    inner2 = routines[1]
    assert 'x' in (var.name for var in FindVariables().visit(inner1.spec))
    assert 'x' in (var.name for var in inner1.arguments)
    assert 'gx' in (var.name for var in FindVariables().visit(inner2.spec))
    assert 'gx' in (var.name for var in inner2.arguments)

    call = [call for call in FindNodes(CallStatement).visit(outer.body) if call.name == "inner1"][0]
    assert 'x' in (var.name for var in call.arguments)
    call = [call for call in FindNodes(CallStatement).visit(outer.body) if call.name == "inner2"][0]
    assert 'gx' in (var.name for var in call.arguments)

def test_extract_contained_procedures_basic_scalar_function():
    """
    Basic test for scalars highlighting that the inner procedure may also be a function. 
    """
    fcode = """
        subroutine outer()
            implicit none
            integer :: x
            integer :: y
            x = 42
            y = inner()
            contains
            function inner() result(z)
                integer :: y
                integer :: z
                y = 1
                z = x + y
            end function inner
        end subroutine outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    assert routines[0].name == "inner"
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'x' in (var.name for var in inner.arguments)

    call = list(FindInlineCalls().visit(outer.body))[0]
    assert 'x' in (var.name for var in call.parameters)

def test_extract_contained_procedures_basic_scalar_function_both():
    """
    Basic test for scalars highlighting that the outer and inner procedure may be functions. 
    """
    fcode = """
        function outer() result(outer_res)
            implicit none
            integer :: x
            integer :: outer_res
            x = 42
            outer_res = inner()
            contains
            function inner() result(z)
                integer :: y
                integer :: z
                y = 1
                z = x + y
            end function inner
        end function outer 
    """
    src = Sourcefile.from_source(fcode)
    routines = extract_contained_procedures(src.routines[0])
    assert len(routines) == 1
    assert routines[0].name == "inner"
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in (var.name for var in FindVariables().visit(inner.spec))
    assert 'x' in (var.name for var in inner.arguments)

    call = list(FindInlineCalls().visit(outer.body))[0]
    assert 'x' in (var.name for var in call.parameters)
