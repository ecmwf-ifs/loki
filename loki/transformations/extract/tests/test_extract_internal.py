# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki.frontend import available_frontends, OMNI
from loki.ir import CallStatement, Import, FindNodes, FindInlineCalls
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine

from loki.transformations.extract import extract_internal_procedures


@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_basic_scalar(frontend):
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 1
    assert routines[0].name == "inner"
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in inner.arguments

    call = FindNodes(CallStatement).visit(outer.body)[0]
    assert 'x' in (arg[0] for arg in call.kwarguments)

@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_contains_emptied(frontend):
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    outer = src.routines[0]
    extract_internal_procedures(outer)
    # NOTE: Functions in Loki are also typed as Subroutines.
    assert not any(isinstance(r, Subroutine) for r in outer.contains.body)

@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_basic_array(frontend):
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 1
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in inner.arguments
    assert 'arr(3)' in inner.arguments

    call = FindNodes(CallStatement).visit(outer.body)[0]
    kwargdict = dict(call.kwarguments)
    assert kwargdict['x'] == 'x'
    assert kwargdict['arr'] == 'arr'

@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_existing_call_args(frontend):
    """
    Tests that variable resolution process works correctly when the parent contains a call to
    the extracted function that already has some calling arguments.
    Test also that new args are introduced as kw arguments.
    """

    fcode = """
        subroutine outer()
            implicit none
            integer :: x
            integer :: y
            integer :: z
            real :: arr(3)
            arr = 71.0
            x = 42
            y = 1
            call inner(x, y)
            call inner(x, y = 1)
            ! Note, 'call inner(y = 1, x)' is disallowed by Fortran and not tested.
            call inner(x = 1, y = 1)
            call inner(y = 1, x = 1)
            contains
            subroutine inner(x, y)
                integer, intent(in) :: x
                integer, intent(in) :: y
                z = x + y + arr(1)
            end subroutine inner
        end subroutine outer
    """
    src = Sourcefile.from_source(fcode, frontend=frontend)
    outer = src.routines[0]
    extract_internal_procedures(outer)
    calls = FindNodes(CallStatement).visit(outer.body)

    for call in calls:
        kwargdict = dict(call.kwarguments)
        assert kwargdict['arr'] == 'arr'
        assert kwargdict['z'] == 'z'

    assert 'x' == calls[0].arguments[0]
    assert 'y' == calls[0].arguments[1]
    assert len(calls[0].arguments) == 2

    assert 'x' == calls[1].arguments[0]
    assert len(calls[1].arguments) == 1
    assert 'y' in tuple(arg[0] for arg in calls[1].kwarguments)

    assert len(calls[2].arguments) == 0
    kwargdict = dict(calls[2].kwarguments)
    assert kwargdict['x'] == 1
    assert kwargdict['y'] == 1

    assert len(calls[3].arguments) == 0
    kwargdict = dict(calls[3].kwarguments)
    assert kwargdict['x'] == 1
    assert kwargdict['y'] == 1

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails on missing constants module')]))
def test_extract_internal_procedures_basic_import(frontend):
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 1
    inner = routines[0]
    assert "c2" in inner.import_map
    assert "c1" not in inner.import_map
    assert 'c2' not in inner.arguments

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails on missing type_mod module')]))
def test_extract_internal_procedures_recursive_definition(frontend):
    """
    Tests that whenever a global in the contained subroutine depends on another
    global variable, both are introduced as arguments,
    even if there is no explicit reference to the latter.
    """
    fcode = """
        subroutine outer(klon, klev, mt)
            use type_mod, only: mytype
            implicit none
            integer, intent(in) :: klon
            integer, intent(in) :: klev
            type(mytype), intent(in) :: mt
            integer :: somearr(klon, mt%a%b)
            integer :: x(klon)
            integer :: somevar(klon, klev + 1)

            x(klon - 1) = 42
            call inner()
            contains
            subroutine inner()
                integer :: y
                integer :: z
                y = 1
                z = x(1) + y + somevar(1, 1) - somearr(1, 1)
            end subroutine inner
        end subroutine outer
    """
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 1
    outer = src.routines[0]
    inner = routines[0]
    assert 'x(klon)' in inner.arguments
    assert 'somevar(klon, klev + 1)' in inner.arguments
    assert 'klon' in inner.arguments
    assert 'klev' in inner.arguments
    assert 'mt' in inner.arguments
    assert 'mt%a' not in inner.arguments
    assert 'mt%a%b' not in inner.arguments

    call = FindNodes(CallStatement).visit(outer.body)[0]
    kwargdict = dict(call.kwarguments)
    assert kwargdict['x'] == 'x'
    assert kwargdict['klon'] == 'klon'
    assert kwargdict['somearr'] == 'somearr'
    assert kwargdict['somevar'] == 'somevar'
    assert kwargdict['klev'] == 'klev'
    assert kwargdict['mt'] == 'mt'
    assert 'mt%a' not in kwargdict
    assert 'mt%a%b' not in kwargdict

    assert 'x' not in call.arguments
    assert 'klon' not in call.arguments
    assert 'somearr' not in call.arguments
    assert 'somevar' not in call.arguments
    assert 'klev' not in call.arguments
    assert 'mt' not in call.arguments
    assert 'mt%a' not in call.arguments
    assert 'mt%a%b' not in call.arguments

    # Test that intent of 'klon' and 'klev' is also 'in' inside inner (because intent is given in parent).
    klon = inner.variable_map['klon']
    klev = inner.variable_map['klev']
    assert klon.type.intent == "in"
    assert klev.type.intent == "in"

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails on missing parkind1 module')]))
def test_extract_internal_procedures_recursive_definition_import(frontend):
    """
    Tests that whenever globals in the contained subroutine depend on imported bindings,
    the globals are introduced as arguments, and the imports are added to the contained subroutine.
    """
    fcode = """
        subroutine outer()
            use parkind1, only: jprb, jpim
            implicit none
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 1
    outer = src.routines[0]
    inner = routines[0]
    assert 'x(3)' in inner.arguments
    assert 'ii(30)' in inner.arguments
    call = FindNodes(CallStatement).visit(outer.body)[0]
    kwargdict = dict(call.kwarguments)
    assert kwargdict['x'] == 'x'
    assert kwargdict['ii'] == 'ii'

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

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails on missing parkind1 module')]))
def test_extract_internal_procedures_kind_resolution(frontend):
    """
    Tests that an unresolved kind parameter in inner scope is resolved from import in outer scope.
    """
    fcode = """
        subroutine outer()
            use parkind1, only: jpim
            implicit none
            call inner()
            contains
            subroutine inner()
                integer(kind = jpim) :: y
                integer(kind=8) :: z
                z = y
            end subroutine inner
        end subroutine outer
    """
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    inner = routines[0]
    assert "jpim" in inner.import_map

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails on missing stuff module')]))
def test_extract_internal_procedures_derived_type_resolution(frontend):
    """
    Tests that an unresolved derived type in inner scope is resolved from import in outer scope.
    """
    fcode = """
        subroutine outer()
            use stuff, only: mytype
            implicit none
            call inner()
            contains
            subroutine inner()
                type(mytype) :: y
                integer :: z
                z = y%a
            end subroutine inner
        end subroutine outer
    """
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    inner = routines[0]
    assert "mytype" in inner.import_map

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails on missing types module')]))
def test_extract_internal_procedures_derived_type_field(frontend):
    """
    Test that when a derived type field, i.e 'a%b' is a global in the scope of the contained subroutine,
    the derived type itself, that is, 'a', is introduced as an the argument in the transformation.
    """
    fcode = """
        subroutine outer()
            use types, only: my_type, your_type
            implicit none
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    outer = src.routines[0]
    inner = routines[0]
    assert 'xtyp' in inner.arguments
    assert 'ytyp' in inner.arguments

    call = FindNodes(CallStatement).visit(outer.body)[0]
    kwargdict = dict(call.kwarguments)
    assert kwargdict['xtyp'] == 'xtyp'
    assert kwargdict['ytyp'] == 'ytyp'

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

@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_intent(frontend):
    """
    This test is just to document the current behaviour: when a global is
    introduced as an argument to the extracted contained procedure,
    its intent will be 'inout', unless the intent is specified in the parent procedure.
    """
    fcode = """
        subroutine outer(v, p)
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
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

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails on undefined symbols')]))
def test_extract_internal_procedures_undefined_in_parent(frontend):
    """
    This test is just to document current behaviour:
    an exception is raised if a global inside the contained procedure does not
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    with pytest.raises(RuntimeError):
        extract_internal_procedures(src.routines[0])

@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_multiple_internal_procedures(frontend):
    """
    Basic test to check that multiple contained procedures can also be handled.
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 2
    assert routines[0].name == "inner1"
    assert routines[1].name == "inner2"
    outer = src.routines[0]
    inner1 = routines[0]
    inner2 = routines[1]
    assert 'x' in inner1.arguments
    assert 'gx' in inner2.arguments

    call = [call for call in FindNodes(CallStatement).visit(outer.body) if call.name == "inner1"][0]
    assert 'x' in (arg[0] for arg in call.kwarguments)
    call = [call for call in FindNodes(CallStatement).visit(outer.body) if call.name == "inner2"][0]
    assert 'gx' in (arg[0] for arg in call.kwarguments)

@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_basic_scalar_function(frontend):
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 1
    assert routines[0].name == "inner"
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in inner.arguments

    call = list(FindInlineCalls().visit(outer.body))[0]
    assert 'x' in call.kw_parameters

@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_internal_procedures_basic_scalar_function_both(frontend):
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
    src = Sourcefile.from_source(fcode, frontend=frontend)
    routines = extract_internal_procedures(src.routines[0])
    assert len(routines) == 1
    assert routines[0].name == "inner"
    inner = routines[0]
    outer = src.routines[0]
    assert 'x' in inner.arguments

    call = list(FindInlineCalls().visit(outer.body))[0]
    assert 'x' in call.kw_parameters
