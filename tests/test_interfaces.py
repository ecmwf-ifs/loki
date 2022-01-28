from pathlib import Path
import pytest

from conftest import available_frontends
from loki import Module, Subroutine, FindNodes, Interface, Import, fgen, OMNI


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_spec(frontend):
    """
    Test basic functionality of interface representation
    """
    fcode = """
module interface_spec_mod
    interface
        subroutine sub(a, b)
            integer, intent(in) :: a
            integer, intent(out) :: b
        end subroutine sub
    end interface
end module interface_spec_mod
    """.strip()

    # Parse the source and find the interface
    module = Module.from_source(fcode, frontend=frontend)
    interfaces = FindNodes(Interface).visit(module.spec)
    assert len(interfaces) == 1
    interface = interfaces[0]

    # Make sure basic properties are right
    assert interface.abstract is False
    assert interface.symbols == ('sub',)

    # Check the subroutine is there
    assert len(interface.body) == 1
    assert isinstance(interface.body[0], Subroutine)

    # Sanity check fgen
    code = module.to_fortran().lower()
    assert 'interface' in code
    assert 'end interface' in code
    assert 'subroutine sub' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_module_integration(frontend):
    """
    Test correct integration of interfaces into modules
    """
    fcode = """
module interface_module_integration_mod
    abstract interface
        subroutine sub(a, b)
            integer, intent(in) :: a
            integer, intent(out) :: b
        end subroutine sub
    end interface
end module interface_module_integration_mod
    """.strip()

    # Parse the source and find the interface
    module = Module.from_source(fcode, frontend=frontend)
    assert len(module.interfaces) == 1
    interface = module.interfaces[0]
    assert isinstance(interface, Interface)

    # Make sure declared symbols are accessible through various properties
    assert interface.symbols == ('sub',)
    assert module.interface_symbols == ('sub',)
    assert module.interface_map['sub'] is interface
    assert module.interface_symbol_map == {'sub': interface.symbols[0]}
    assert 'sub' in module.symbols
    assert module.symbol_map['sub'] == interface.symbols[0]

    # Sanity check fgen
    code = module.to_fortran().lower()
    assert 'abstract interface' in code
    assert 'subroutine sub' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_subroutine_integration(frontend):
    """
    Test correct integration of interfaces into subroutines
    """
    fcode = """
subroutine interface_subroutine_integration(X, Y, N, PROC)
    INTEGER, INTENT(IN) :: X(:,:), N
    INTEGER, INTENT(OUT) :: Y(:)
    INTERFACE
        SUBROUTINE PROC(A, B)
            INTEGER, INTENT(IN) :: A(:)
            INTEGER, INTENT(OUT) :: B
        END SUBROUTINE PROC
    END INTERFACE
    INTEGER :: I

    DO I=1,N
        CALL PROC(X(:, I), Y(I))
    END DO
end subroutine interface_subroutine_integration
    """.strip()

    # Parse the source and find the interface
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(routine.interfaces) == 1
    interface = routine.interfaces[0]
    assert isinstance(interface, Interface)

    # Make sure the declared symbols are accessible through various properties
    assert interface.symbols == ('proc',)
    assert routine.interface_symbols == ('proc',)
    assert routine.interface_map['proc'] is interface
    assert routine.interface_symbol_map == {'proc': interface.symbols[0]}
    assert 'proc' in routine.symbols
    assert routine.symbol_map['proc'] == interface.symbols[0]
    assert 'proc' in routine.arguments
    assert 'proc' in [arg.lower() for arg in routine.argnames]

    # Sanity check fgen
    code = routine.to_fortran().lower()
    assert 'interface' in code
    assert 'subroutine proc' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_import(frontend):
    """
    Test correct representation of ``IMPORT`` statements in interfaces
    """
    # Example from F2008, Note 12.5
    # The IMPORT statement can be used to allow module procedures to have dummy arguments that are
    # procedures with assumed-shape arguments of an opaque type.
    # The MONITOR dummy procedure requires an explicit interface because it has an assumed-shape array
    # argument, but TYPE(T) would not be available inside the interface body without the IMPORT statement.
    fcode = """
module interface_import_mod
    TYPE T
        PRIVATE ! T is an opaque type
    END TYPE
CONTAINS
    SUBROUTINE PROCESS(X,Y,RESULT,MONITOR)
        TYPE(T),INTENT(IN) :: X(:,:),Y(:,:)
        TYPE(T),INTENT(OUT) :: RESULT(:,:)
        INTERFACE
            SUBROUTINE MONITOR(ITERATION_NUMBER,CURRENT_ESTIMATE)
                IMPORT T
                INTEGER,INTENT(IN) :: ITERATION_NUMBER
                TYPE(T),INTENT(IN) :: CURRENT_ESTIMATE(:,:)
            END SUBROUTINE
        END INTERFACE
    END SUBROUTINE
end module interface_import_mod
    """.strip()

    # Parse the source and find the interface
    module = Module.from_source(fcode, frontend=frontend)
    interface = module['process'].interface_map['monitor']

    # Find the import statement and test its properties
    assert len(interface.body) == 1
    imprts = FindNodes(Import).visit(interface.body[0].spec)
    assert len(imprts) == 1

    # Sanity check fgen
    assert fgen(imprts[0]).lower() == 'import t'
    assert 'import t' in fgen(interface).lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_multiple_routines(frontend):
    """
    Test interfaces with multiple subroutine/function declarations
    in the interface block
    """
    # Example from F2008, Note 12.4
    fcode = """
module interface_multiple_routines_mod
    INTERFACE
        SUBROUTINE EXT1 (X, Y, Z)
            REAL, DIMENSION (100, 100) :: X, Y, Z
        END SUBROUTINE EXT1
        SUBROUTINE EXT2 (X, Z)
            REAL X
            COMPLEX (KIND = 4) Z (2000)
        END SUBROUTINE EXT2
        FUNCTION EXT3 (P, Q)
            LOGICAL EXT3
            INTEGER P (1000)
            LOGICAL Q (1000)
        END FUNCTION EXT3
    END INTERFACE
end module interface_multiple_routines_mod
    """.strip()

    # Parse the source and find the interface
    module = Module.from_source(fcode, frontend=frontend)

    if frontend == OMNI:
        # OMNI has to do things differently, of course, and splits the interface
        # block into separate blocks for each procedures
        assert len(module.interfaces) == 3

        # Make sure interfaces can be found under their declared names
        for intf in module.interfaces:
            assert len(intf.symbols) == 1
            name = str(intf.symbols[0])
            assert module.interface_map[name] is intf

    else:
        assert len(module.interfaces) == 1
        intf = module.interfaces[0]

        # Make sure interface is found under all names
        assert all(module.interface_map[name] is intf for name in ['ext1', 'ext2', 'ext3'])

        # Make sure declared names end up in the right places
        assert intf.symbols == ('ext1', 'ext2', 'ext3')

    assert all(name in module.symbols for name in ('ext1', 'ext2', 'ext3'))

    # Sanity check fgen
    code = module.to_fortran().lower()
    assert 'subroutine ext1' in code
    assert 'subroutine ext2' in code
    assert 'function ext3' in code
