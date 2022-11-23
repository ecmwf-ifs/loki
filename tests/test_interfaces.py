# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from conftest import available_frontends
from loki import (
    Module, Subroutine, FindNodes, Interface, Import, fgen, OMNI,
    ProcedureSymbol, ProcedureType
)


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
    assert routine.symbol_map['proc'].type.dtype.procedure is interface.body[0]

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


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_generic_spec(frontend):
    """
    Test interfaces with a generic identifier
    """
    # Fortran 2008, Note 12.6
    fcode = """
module interface_generic_spec_mod
    IMPLICIT NONE
    INTERFACE SWITCH
        SUBROUTINE INT_SWITCH (X, Y)
        INTEGER, INTENT (INOUT) :: X, Y
        END SUBROUTINE INT_SWITCH
        SUBROUTINE REAL_SWITCH (X, Y)
            REAL, INTENT (INOUT) :: X, Y
        END SUBROUTINE REAL_SWITCH
        SUBROUTINE COMPLEX_SWITCH (X, Y)
            COMPLEX, INTENT (INOUT) :: X, Y
        END SUBROUTINE COMPLEX_SWITCH
    END INTERFACE SWITCH
end module interface_generic_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    if frontend == OMNI:
        # FANTASTIC... OMNI helps us to a treat and separates the subroutine interfaces
        # from the generic interface...
        assert len(module.interfaces) == 4
    else:
        assert len(module.interfaces) == 1

    assert set(module.interfaces[-1].symbols) == {'switch', 'int_switch', 'real_switch', 'complex_switch'}

    # This applies only to OMNI
    for intf in module.interfaces[:-1]:
        assert intf.spec is None

    # Now the actual generic interface
    intf = module.interfaces[-1]
    assert isinstance(intf.spec, ProcedureSymbol)
    assert intf.spec.scope is module
    assert intf.spec == 'switch'
    assert intf.spec.type.dtype.is_generic is True
    assert 'INTERFACE SWITCH' in fgen(intf).upper()

    assert all(s in module.symbols for s in ('switch', 'int_switch', 'real_switch', 'complex_switch'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_operator_module_procedure(frontend):
    """
    Test interfaces that declare generic operators and refer to module procedures
    """
    fcode = """
MODULE SPECTRAL_FIELDS_MOD
IMPLICIT NONE
PRIVATE
PUBLIC SPECTRAL_FIELD, ASSIGNMENT(=), OPERATOR(.EQV.)

! Trimmed-down version !
TYPE SPECTRAL_FIELD
    REAL, ALLOCATABLE :: SP2D(:,:)
    INTEGER :: NS2D
    INTEGER :: NSPEC2
END TYPE SPECTRAL_FIELD

INTERFACE ASSIGNMENT (=)
    MODULE PROCEDURE ASSIGN_SCALAR_SP, ASSIGN_SP_AR
END INTERFACE

INTERFACE OPERATOR (.EQV.)
    PROCEDURE EQUIV_SPEC
END INTERFACE

CONTAINS

SUBROUTINE ASSIGN_SCALAR_SP(YDSP,PVAL)
    TYPE (SPECTRAL_FIELD), INTENT(INOUT) :: YDSP
    REAL, INTENT(IN) :: PVAL
    YDSP%SP2D(:,:)  =PVAL
END SUBROUTINE ASSIGN_SCALAR_SP

SUBROUTINE ASSIGN_SP_AR(PFLAT,YDSP)
    REAL, INTENT(OUT) :: PFLAT(:)
    TYPE (SPECTRAL_FIELD), INTENT(IN) :: YDSP
    INTEGER :: I2D,ISHAPE2D(1)

    I2D=YDSP%NS2D*YDSP%NSPEC2
    ISHAPE2D(1)=I2D
    PFLAT(    1:    I2D)=RESHAPE(YDSP%SP2D(:,:)  ,ISHAPE2D)
END SUBROUTINE ASSIGN_SP_AR

LOGICAL FUNCTION EQUIV_SPEC(YDSP1,YDSP2)
    TYPE(SPECTRAL_FIELD), INTENT(IN) :: YDSP1
    TYPE(SPECTRAL_FIELD), INTENT(IN) :: YDSP2
    LOGICAL :: LL
    INTEGER :: JF, JM
    ! Modified for simplicity!

    LL = .TRUE.
    LL = LL .AND. (YDSP1%NS2D ==YDSP2%NS2D)
    LL = LL .AND. (YDSP1%NSPEC2 ==YDSP2%NSPEC2)
    IF (LL) THEN
        DO JF=1,YDSP1%NS2D
            DO JM=1,YDSP1%NSPEC2
                LL = LL .AND. (YDSP1%SP2D(JF, JM)==YDSP2%SP2D(JF, JM))
            ENDDO
        ENDDO
    ENDIF

    EQUIV_SPEC=LL
END FUNCTION EQUIV_SPEC
END MODULE SPECTRAL_FIELDS_MOD
    """.strip()

    mod = Module.from_source(fcode, frontend=frontend)
    assert len(mod.interfaces) == 2

    assign_intf = mod.interface_map['assignment(=)']
    assert assign_intf.spec == 'assignment(=)'
    assert set(assign_intf.symbols) == {'assignment(=)', 'assign_scalar_sp', 'assign_sp_ar'}

    assign_map = {s.name.lower(): s for s in assign_intf.symbols}
    assert assign_map['assignment(=)'].type.dtype.is_generic is True
    assert assign_map['assign_scalar_sp'].type.dtype.procedure is mod['assign_scalar_sp']
    assert assign_map['assign_scalar_sp'].type.dtype.is_generic is False
    assert assign_map['assign_sp_ar'].type.dtype.procedure is mod['assign_sp_ar']
    assert assign_map['assign_sp_ar'].type.dtype.is_generic is False

    if frontend == OMNI:  # One declaration per line... :eyeroll:
        assert len(assign_intf.body) == 2
    else:
        assert len(assign_intf.body) == 1
    assign_decl = assign_intf.body[0]
    assert assign_decl.module is True

    op_intf = mod.interface_map['operator(.eqv.)']
    assert op_intf.spec == 'operator(.eqv.)'
    assert set(op_intf.symbols) == {'operator(.eqv.)', 'equiv_spec'}

    op_map = {s.name.lower(): s for s in op_intf.symbols}
    assert op_map['operator(.eqv.)'].type.dtype.is_generic is True
    assert op_map['equiv_spec'].type.dtype.procedure is mod['equiv_spec']
    assert op_map['equiv_spec'].type.dtype.is_generic is False

    assert len(op_intf.body) == 1
    op_decl = op_intf.body[0]

    if frontend != OMNI:  # Grrr...
        assert op_decl.module is False

    assign_code = fgen(assign_intf).lower().strip()
    assert assign_code.startswith('interface assignment(=)')
    assert assign_code.endswith('end interface assignment(=)')
    assert 'module procedure' in assign_code

    op_code = fgen(op_intf).lower().strip()
    assert op_code.startswith('interface operator(.eqv.)')
    assert op_code.endswith('end interface operator(.eqv.)')

    if frontend != OMNI:  # *...*
        assert 'module' not in op_code

    other_code = """
module use_spectral_fields_mod
    use spectral_fields_mod, only: assignment(=), operator(.eqv.)
end module use_spectral_fields_mod
    """.strip()

    other_mod = Module.from_source(other_code, frontend=frontend, definitions=[mod])
    assert set(other_mod.symbols) == {'assignment(=)', 'operator(.eqv.)'}
    assert other_mod.imported_symbols == ('assignment(=)', 'operator(.eqv.)')

    assign_sym = other_mod.imported_symbol_map['assignment(=)']
    assert isinstance(assign_sym, ProcedureSymbol)
    assert isinstance(assign_sym.type.dtype, ProcedureType)
    assert assign_sym.type.dtype.is_generic is True
    assert assign_sym.type.imported is True
    assert assign_sym.type.module is mod

    op_sym = other_mod.imported_symbol_map['operator(.eqv.)']
    assert isinstance(op_sym, ProcedureSymbol)
    assert isinstance(op_sym.type.dtype, ProcedureType)
    assert op_sym.type.dtype.is_generic is True
    assert op_sym.type.imported is True
    assert op_sym.type.module is mod

    assert other_code.splitlines()[1].strip() in fgen(other_mod).lower()
