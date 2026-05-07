# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Subroutine, fgen
from loki.jit_build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes, FindVariables, Section, SubstituteExpressions

from loki.transformations.array_indexing.promote import promote_variables, promote_variable_declarations
from loki.transformations.array_indexing.promote_local_array import PromoteLocalArrayTransformation
from loki.transformations.utilities import update_variable_declarations
from loki.dimension import Dimension


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variable_scalar(tmp_path, frontend):
    """
    Apply variable promotion for a single scalar variable.
    """
    fcode = """
subroutine transform_promote_variable_scalar(ret)
  implicit none
  integer, intent(out) :: ret
  integer :: tmp, jk

  ret = 0
  do jk=1,10
    tmp = jk
    ret = ret + tmp
  end do
end subroutine transform_promote_variable_scalar
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    ret = function()
    assert ret == 55

    # Apply and test the transformation
    assert isinstance(routine.variable_map['tmp'], sym.Scalar)
    promote_variables(routine, ['TMP'], pos=0, index=routine.variable_map['JK'], size=sym.Literal(10))
    assert isinstance(routine.variable_map['tmp'], sym.Array)
    assert routine.variable_map['tmp'].shape == (sym.Literal(10),)

    promoted_filepath = tmp_path/(f'{routine.name}_promoted_{frontend}.f90')
    promoted_function = jit_compile(routine, filepath=promoted_filepath, objname=routine.name)
    ret = promoted_function()
    assert ret == 55


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variables(tmp_path, frontend):
    """
    Apply variable promotion for scalar and array variables.
    """
    fcode = """
subroutine transform_promote_variables(scalar, vector, n)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: scalar, vector(n)
  integer :: tmp_scalar, tmp_vector(n), tmp_matrix(n,n)
  integer :: jl, jk

  do jl=1,n
    ! a bit of a hack to create initialized meaningful output
    tmp_vector(:) = 0
  end do

  do jl=1,n
    tmp_scalar = jl
    tmp_vector(jl) = jl

    do jk=1,n
      tmp_matrix(jk, jl) = jl + jk
    end do
  end do

  scalar = 0
  do jl=1,n
    scalar = scalar + tmp_scalar
    vector = tmp_matrix(:,jl) + tmp_vector(:)
  end do
end subroutine transform_promote_variables
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 10
    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    function(scalar, vector, n)
    assert scalar == n*n
    assert np.all(vector == np.array(list(range(1, 2*n+1, 2)), order='F', dtype=np.int32) + n + 1)

    # Verify dimensions before promotion
    assert isinstance(routine.variable_map['tmp_scalar'], sym.Scalar)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])

    # Promote scalar and vector and verify dimensions
    promote_variables(routine, ['tmp_scalar', 'tmp_vector'], pos=-1, index=routine.variable_map['JL'],
                      size=routine.variable_map['n'])

    assert isinstance(routine.variable_map['tmp_scalar'], sym.Array)
    assert routine.variable_map['tmp_scalar'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'], routine.variable_map['n'])
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])

    # Promote matrix and verify dimensions
    promote_variables(routine, ['tmp_matrix'], pos=1, index=routine.variable_map['JL'],
                      size=routine.variable_map['n'])

    assert isinstance(routine.variable_map['tmp_scalar'], sym.Array)
    assert routine.variable_map['tmp_scalar'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'], routine.variable_map['n'])
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], ) * 3

    # Test promoted routine
    promoted_filepath = tmp_path/(f'{routine.name}_promoted_{frontend}.f90')
    promoted_function = jit_compile(routine, filepath=promoted_filepath, objname=routine.name)

    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    promoted_function(scalar, vector, n)
    assert scalar == n*(n+1)//2
    assert np.all(vector[:-1] == np.array(list(range(n + 1, 2*n)), order='F', dtype=np.int32))
    assert vector[-1] == 3*n


@pytest.mark.parametrize('frontend', available_frontends())
def test_update_variable_declarations(frontend):
    """
    Test that :any:`update_variable_declarations` updates DIMENSION(...)
    attributes on declarations that still retain the DIMENSION keyword
    after promotion.

    When a variable is the sole symbol in a ``DIMENSION(...)`` declaration,
    ``single_variable_declaration`` (called by ``promote_variables``) converts
    it to inline dimensions (``dimensions=None`` on the decl). When multiple
    variables share a ``DIMENSION(...)`` declaration and all are promoted,
    the DIMENSION attribute is updated to reflect the new shape.
    """
    fcode = """
subroutine test_update_decl(n, m)
  implicit none
  integer, intent(in) :: n, m
  integer, dimension(n) :: a, c
  integer, dimension(n, m) :: b

  a(1) = 1
  c(1) = 3
  b(1, 1) = 2
end subroutine test_update_decl
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Verify original declarations have DIMENSION set
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    ac_decl = [d for d in decls if any(s.name.lower() == 'a' for s in d.symbols)][0]
    b_decl = [d for d in decls if any(s.name.lower() == 'b' for s in d.symbols)][0]
    assert ac_decl.dimensions is not None
    assert len(ac_decl.dimensions) == 1
    assert any(s.name.lower() == 'c' for s in ac_decl.symbols)
    assert b_decl.dimensions is not None
    assert len(b_decl.dimensions) == 2

    # Promote both 'a' and 'c' (all variables in that DIMENSION decl)
    promote_variables(routine, ['a', 'c'], pos=-1, size=sym.Literal(10))

    # After promotion, single_variable_declaration splits the shared decl
    # into individual ones with dimensions=None (inline dims on each symbol).
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    a_decl = [d for d in decls if any(s.name.lower() == 'a' for s in d.symbols)][0]
    c_decl = [d for d in decls if any(s.name.lower() == 'c' for s in d.symbols)][0]

    # Verify variable shapes are correctly promoted
    assert routine.variable_map['a'].shape == (routine.variable_map['n'], sym.Literal(10))
    assert routine.variable_map['c'].shape == (routine.variable_map['n'], sym.Literal(10))

    # Verify a and c are in separate declarations (split by single_variable_declaration)
    assert a_decl is not c_decl

    # Verify b is unchanged
    b_decl = [d for d in decls if any(s.name.lower() == 'b' for s in d.symbols)][0]
    assert b_decl.dimensions is not None
    assert len(b_decl.dimensions) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variable_dimension_attribute(frontend):
    """
    Test promotion when variables are declared with shared ``DIMENSION(...)``
    keyword and only a subset are promoted.

    This is the key regression test: when two variables share a declaration
    like ``INTEGER, DIMENSION(n) :: a, b`` and only ``a`` is promoted, the
    declaration must be split and the promoted variable's declaration must
    have the correct new shape.
    """
    fcode = """
subroutine transform_promote_dim_attr(ret, n)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: ret
  integer, dimension(n) :: vec1, vec2
  integer :: jl, jk

  do jk=1,10
    do jl=1,n
      vec1(jl) = jl + jk
      vec2(jl) = jl * 2
    end do
  end do

  ret = 0
  do jl=1,n
    ret = ret + vec1(jl) + vec2(jl)
  end do
end subroutine transform_promote_dim_attr
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Verify both variables share a DIMENSION declaration
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    dim_decls = [d for d in decls if d.dimensions is not None]
    shared_decl = [d for d in dim_decls
                   if any(s.name.lower() == 'vec1' for s in d.symbols)
                   and any(s.name.lower() == 'vec2' for s in d.symbols)]
    assert len(shared_decl) == 1, "vec1 and vec2 should share a DIMENSION declaration"

    # Promote only vec1 with a new trailing dimension of size 10
    promote_variables(routine, ['vec1'], pos=-1, index=routine.variable_map['jk'],
                      size=sym.Literal(10))

    # vec1 should now be a 2D array
    assert isinstance(routine.variable_map['vec1'], sym.Array)
    assert routine.variable_map['vec1'].shape == (routine.variable_map['n'], sym.Literal(10))

    # vec2 should still be a 1D array
    assert isinstance(routine.variable_map['vec2'], sym.Array)
    assert routine.variable_map['vec2'].shape == (routine.variable_map['n'],)

    # After promotion with single_variable_declaration, vec1 and vec2
    # should be in separate declarations
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    vec1_decls = [d for d in decls if any(s.name.lower() == 'vec1' for s in d.symbols)]
    vec2_decls = [d for d in decls if any(s.name.lower() == 'vec2' for s in d.symbols)]
    assert len(vec1_decls) == 1
    assert len(vec2_decls) == 1
    assert vec1_decls[0] is not vec2_decls[0], "Declarations should have been split"


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variable_scalar_with_size(frontend):
    """
    Promote a scalar variable with only size (no index), verifying that
    the declaration is updated correctly.
    """
    fcode = """
subroutine transform_promote_scalar_size(n)
  implicit none
  integer, intent(in) :: n
  integer :: tmp

  tmp = n + 1
end subroutine transform_promote_scalar_size
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Promote a scalar with an explicit size only
    assert isinstance(routine.variable_map['tmp'], sym.Scalar)

    promote_variables(routine, ['tmp'], pos=0, size=sym.Literal(10))

    assert isinstance(routine.variable_map['tmp'], sym.Array)
    assert routine.variable_map['tmp'].shape == (sym.Literal(10),)

    # After promoting a scalar (no original DIMENSION), the declaration should
    # use inline dimensions on the variable itself
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    tmp_decl = [d for d in decls if any(s.name.lower() == 'tmp' for s in d.symbols)][0]
    assert isinstance(tmp_decl.symbols[0], sym.Array)
    assert tmp_decl.symbols[0].shape == (sym.Literal(10),)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variable_with_range_index_size(frontend):
    """
    Promote a local array variable using a :any:`RangeIndex` as the size,
    mimicking the pattern used by :any:`PromoteLocalArrayTransformation`.
    """
    fcode = """
subroutine transform_promote_range_size(ret, n, istart, iend)
  implicit none
  integer, intent(in) :: n, istart, iend
  integer, intent(out) :: ret
  integer :: tmp(n), jl, jcol

  do jcol=istart,iend
    do jl=1,n
      tmp(jl) = jl + jcol
    end do
  end do

  ret = 0
  do jl=1,n
    ret = ret + tmp(jl)
  end do
end subroutine transform_promote_range_size
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    lower = routine.variable_map['istart']
    upper = routine.variable_map['iend']
    jcol = routine.variable_map['jcol']

    assert isinstance(routine.variable_map['tmp'], sym.Array)
    assert routine.variable_map['tmp'].shape == (routine.variable_map['n'],)

    # Promote tmp with a RangeIndex size, like PromoteLocalArrayTransformation does
    promote_variables(routine, ['tmp'], pos=-1,
                      index=jcol, size=sym.RangeIndex((lower, upper)))

    # Verify promoted shape
    assert isinstance(routine.variable_map['tmp'], sym.Array)
    assert len(routine.variable_map['tmp'].shape) == 2
    assert routine.variable_map['tmp'].shape[0] == routine.variable_map['n']
    assert isinstance(routine.variable_map['tmp'].shape[1], sym.RangeIndex)

    # Verify the declaration uses the promoted horizontal range
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    tmp_decl = [d for d in decls if any(s.name.lower() == 'tmp' for s in d.symbols)][0]
    assert isinstance(tmp_decl.symbols[0].shape[1], sym.RangeIndex)
    assert tmp_decl.symbols[0].shape[1].lower == lower
    assert tmp_decl.symbols[0].shape[1].upper == upper


@pytest.mark.parametrize('frontend', available_frontends())
def test_update_variable_declarations_mismatched_shapes(frontend):
    """
    Test that :any:`update_variable_declarations` splits a ``DIMENSION(...)``
    declaration into separate declarations when the symbols have different
    shapes (e.g., only one variable in a shared declaration was promoted).

    This exercises the shape-mismatch branch that splits declarations,
    matching the behaviour previously only in :any:`demote_variables`.
    """
    fcode = """
subroutine test_mismatched(n)
  implicit none
  integer, intent(in) :: n
  integer, dimension(n) :: a, b

  a(1) = 1
  b(1) = 2
end subroutine test_mismatched
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    shared_decl = [d for d in decls if d.dimensions is not None
                   and any(s.name.lower() == 'a' for s in d.symbols)
                   and any(s.name.lower() == 'b' for s in d.symbols)]
    assert len(shared_decl) == 1

    # Manually promote only 'a' via SubstituteExpressions to create a shared
    # declaration with mismatched symbol shapes
    a_var = routine.variable_map['a']
    new_shape = (routine.variable_map['n'], sym.Literal(10))
    new_a = a_var.clone(type=a_var.type.clone(shape=new_shape), dimensions=new_shape)
    routine.spec = SubstituteExpressions({a_var: new_a}).visit(routine.spec)

    # update_variable_declarations should detect the mismatch and split
    routine.spec = update_variable_declarations(routine.spec, [new_a])

    # Verify declarations were split
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    a_decls = [d for d in decls if any(s.name.lower() == 'a' for s in d.symbols)]
    b_decls = [d for d in decls if any(s.name.lower() == 'b' for s in d.symbols)]
    assert len(a_decls) == 1
    assert len(b_decls) == 1
    assert a_decls[0] is not b_decls[0], "Mismatched shapes should cause declaration split"

    # Verify individual shapes
    a_sym = a_decls[0].symbols[0]
    assert isinstance(a_sym, sym.Array)
    assert a_sym.shape == (routine.variable_map['n'], sym.Literal(10))

    b_sym = b_decls[0].symbols[0]
    assert isinstance(b_sym, sym.Array)
    assert b_sym.shape == (routine.variable_map['n'],)


@pytest.mark.parametrize('frontend', available_frontends())
def test_promote_variable_declarations_only(frontend):
    """
    Test that :any:`promote_variable_declarations` updates only the
    declarations (spec) and leaves the body untouched.
    """
    fcode = """
subroutine test_promote_decl_only(n)
  implicit none
  integer, intent(in) :: n
  integer :: a(n), b(n, n)
  integer :: jl

  do jl=1,n
    a(jl) = jl
    b(jl, 1) = jl
  end do
end subroutine test_promote_decl_only
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Promote only declarations; body should be unmodified
    promote_variable_declarations(routine, ['a', 'b'], pos=-1, size=sym.Literal(10))

    # Check shapes updated
    assert routine.variable_map['a'].shape == (routine.variable_map['n'], sym.Literal(10))
    assert routine.variable_map['b'].shape == (
        routine.variable_map['n'], routine.variable_map['n'], sym.Literal(10)
    )

    # Check body still has original subscripts with no promoted dimension appended
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    a_assign = next(a for a in assigns if a.lhs.name == 'a')
    b_assign = next(a for a in assigns if a.lhs.name == 'b')
    assert a_assign.lhs.dimensions == (routine.variable_map['jl'],)
    assert b_assign.lhs.dimensions == (routine.variable_map['jl'], 1)


@pytest.mark.parametrize('frontend', available_frontends())
def test_promote_variable_declarations_with_range_index(frontend):
    """
    Test :any:`promote_variable_declarations` with a :any:`RangeIndex` size,
    mimicking the pattern in :any:`PromoteLocalArrayTransformation`.
    """
    fcode = """
subroutine test_promote_decl_range(n, istart, iend)
  implicit none
  integer, intent(in) :: n, istart, iend
  integer :: a(n)

  a(1) = 1
end subroutine test_promote_decl_range
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    lower = routine.variable_map['istart']
    upper = routine.variable_map['iend']

    promote_variable_declarations(routine, ['a'], pos=-1,
                                  size=sym.RangeIndex((lower, upper)))

    assert len(routine.variable_map['a'].shape) == 2
    assert routine.variable_map['a'].shape[0] == routine.variable_map['n']
    assert isinstance(routine.variable_map['a'].shape[1], sym.RangeIndex)

    # Verify the declaration keeps the promoted range bounds
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    a_decl = [d for d in decls if any(s.name.lower() == 'a' for s in d.symbols)][0]
    assert isinstance(a_decl.symbols[0].shape[1], sym.RangeIndex)
    assert a_decl.symbols[0].shape[1].lower == lower
    assert a_decl.symbols[0].shape[1].upper == upper


@pytest.mark.parametrize('frontend', available_frontends())
def test_promote_local_array_transformation_scc_context(frontend):
    """
    Test :any:`PromoteLocalArrayTransformation` in an SCC-like context
    where vector sections have been marked by devector but the horizontal
    loop has been removed.

    This verifies:
    - Declarations are promoted with the horizontal dimension
    - Uses inside vector sections get the horizontal index variable
    - Uses outside vector sections get ':'
    - Arrays not used in vector sections are NOT promoted
    """
    fcode = """
subroutine test_scc_promote(ncol, nlev, istartcol, iendcol, x)
  implicit none
  integer, intent(in) :: ncol, nlev, istartcol, iendcol
  real, intent(inout) :: x(nlev, istartcol:iendcol)
  real :: tmp(nlev)
  real :: unused_arr(nlev)
  integer :: jlev, jcol

  ! Simulate post-devector state: vector section with horizontal loop removed
  tmp(:) = 0.0
  do jlev = 1, nlev
    tmp(jlev) = x(jlev, jcol) * 2.0
    x(jlev, jcol) = tmp(jlev)
  end do

  ! This should not be wrapped — it's outside the vector section
  unused_arr(:) = 1.0
end subroutine test_scc_promote
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Manually wrap the vector computation in a vector_section label,
    # as produced by SCCDevectorTransformation
    body_nodes = list(routine.body.body)
    # The first two statements belong to the vector section; the trailing
    # full-array assignment remains outside of it
    vector_body = tuple(body_nodes[:2])
    non_vector_body = tuple(body_nodes[2:])
    vector_section = Section(body=vector_body, label='vector_section')
    routine.body = routine.body.clone(body=(vector_section,) + non_vector_body)

    horizontal = Dimension(
        name='horizontal', size=['ncol'], index='jcol',
        lower=['istartcol'], upper=['iendcol']
    )

    trafo = PromoteLocalArrayTransformation(horizontal)
    trafo.process_kernel(routine)

    # tmp should be promoted because it is used inside the vector section
    assert len(routine.variable_map['tmp'].shape) == 2
    assert routine.variable_map['tmp'].shape[0] == routine.variable_map['nlev']
    assert isinstance(routine.variable_map['tmp'].shape[1], sym.RangeIndex)
    # unused_arr should not be promoted because it is only used outside it
    assert routine.variable_map['unused_arr'].shape == (routine.variable_map['nlev'],)

    # Inside the vector section, the tmp definition should read x at jcol,
    # while the tmp use in the x assignment still keeps the promoted slice
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    tmp_assign = next(
        a for a in assigns
        if a.lhs.name == 'tmp'
        and any(v.name == 'x' for v in FindVariables(unique=False).visit(a.rhs) if isinstance(v, sym.Array))
    )
    x_assign = next(a for a in assigns if a.lhs.name == 'x')
    tmp_rhs_arrays = [v for v in FindVariables(unique=False).visit(tmp_assign.rhs) if isinstance(v, sym.Array)]
    x_rhs_arrays = [v for v in FindVariables(unique=False).visit(x_assign.rhs) if isinstance(v, sym.Array)]
    assert tmp_assign.lhs.dimensions[0] == routine.variable_map['jlev']
    assert isinstance(tmp_assign.lhs.dimensions[1], sym.RangeIndex)
    assert tmp_rhs_arrays[0].dimensions == (routine.variable_map['jlev'], routine.variable_map['jcol'])
    assert x_rhs_arrays[0].dimensions[0] == routine.variable_map['jlev']
    assert isinstance(x_rhs_arrays[0].dimensions[1], sym.RangeIndex)


@pytest.mark.parametrize('frontend', available_frontends())
def test_promote_local_array_implicit_notation_scc_context(frontend):
    """
    Test :any:`PromoteLocalArrayTransformation` when arrays are used without
    explicit subscripts (implicit full-array notation).

    In Fortran, ``a = 0.0`` where ``a`` is declared as ``a(m,n)`` means
    "assign to the entire array". After promotion adds a new dimension,
    the result should be ``a(:,:,jcol)`` inside vector sections (not just
    ``a(jcol)``), and ``a(:,:,:)`` outside vector sections.

    This also covers the case of passing an array without subscripts to a
    call statement: ``CALL foo(a)`` should become ``CALL foo(a(:,:,jcol))``.
    """
    fcode = """
subroutine test_scc_implicit(ncol, nlev, istartcol, iendcol, ngas, x)
  implicit none
  integer, intent(in) :: ncol, nlev, ngas, istartcol, iendcol
  real, intent(inout) :: x(ngas, nlev, istartcol:iendcol)
  real :: tmp(ngas, nlev)
  integer :: jg, jlev, jcol

  ! Vector section: tmp used without subscripts (implicit notation)
  tmp = 0.0
  do jlev = 1, nlev
    do jg = 1, ngas
      tmp(jg, jlev) = x(jg, jlev, jcol) * 2.0
      x(jg, jlev, jcol) = tmp(jg, jlev)
    end do
  end do
end subroutine test_scc_implicit
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Wrap everything in a vector section, as produced by SCCDevector
    body_nodes = list(routine.body.body)
    vector_section = Section(body=tuple(body_nodes), label='vector_section')
    routine.body = routine.body.clone(body=(vector_section,))

    horizontal = Dimension(
        name='horizontal', size=['ncol'], index='jcol',
        lower=['istartcol'], upper=['iendcol']
    )

    trafo = PromoteLocalArrayTransformation(horizontal)
    trafo.process_kernel(routine)

    assert len(routine.variable_map['tmp'].shape) == 3
    assert routine.variable_map['tmp'].shape[0] == routine.variable_map['ngas']
    assert routine.variable_map['tmp'].shape[1] == routine.variable_map['nlev']
    assert isinstance(routine.variable_map['tmp'].shape[2], sym.RangeIndex)

    # Check the generated form for implicit full-array notation
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    init_assign = next(a for a in assigns if a.lhs.name == 'tmp' and not FindVariables(unique=False).visit(a.rhs))
    tmp_assign = next(a for a in assigns if a.lhs.name == 'tmp' and FindVariables(unique=False).visit(a.rhs))
    assert fgen(init_assign).lower().replace(' ', '') == 'tmp(:,:,jcol)=0.0'
    assert 'tmp(jg,jlev,jcol)' in fgen(tmp_assign).lower().replace(' ', '')


@pytest.mark.parametrize('frontend', available_frontends())
def test_promote_variables_implicit_notation(frontend):
    """
    Test that :any:`promote_variables` correctly handles arrays used
    without explicit subscripts (implicit full-array notation).

    When ``a(n)`` is used as just ``a`` (no subscripts), promoting with
    an index should produce ``a(:,jk)`` not ``a(jk)``.
    """
    fcode = """
subroutine test_promote_implicit(n)
  implicit none
  integer, intent(in) :: n
  integer :: a(n), jk

  a = 0
  do jk = 1, 10
    a(jk) = jk
  end do
end subroutine test_promote_implicit
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    promote_variables(routine, ['a'], pos=-1,
                      index=routine.variable_map['jk'],
                      size=sym.Literal(10))

    # Shape should now be 2D
    assert routine.variable_map['a'].shape == (routine.variable_map['n'], sym.Literal(10))

    # Check the generated form for implicit full-array notation
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert fgen(assigns[0]).lower().replace(' ', '') == 'a(:,:)=0'
    assert fgen(assigns[1]).lower().replace(' ', '') == 'a(jk,jk)=jk'


@pytest.mark.parametrize('frontend', available_frontends())
def test_promote_variable_declarations_deferred_shape(frontend):
    """
    Test that :any:`promote_variable_declarations` keeps the promoted
    dimension deferred (``:``) when the original shape is entirely
    deferred (assumed-shape or allocatable).

    Mixing deferred and explicit dimensions in a declaration is invalid
    Fortran, so ``REAL :: a(:, :, 10)`` must not be produced. Instead
    the result should be ``a(:, :, :)``.

    An explicit-shape variable in the same routine must still get the
    explicit promoted size.
    """
    fcode = """
subroutine test_deferred_promote(n, m)
  implicit none
  integer, intent(in) :: n, m
  real, dimension(:, :) :: a
  real :: b(n, m)

  a(1, 1) = 1.0
  b(1, 1) = 2.0
end subroutine test_deferred_promote
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    promote_variable_declarations(routine, ['a', 'b'], pos=-1, size=sym.Literal(10))

    a_shape = routine.variable_map['a'].shape
    assert len(a_shape) == 3
    for d in a_shape:
        assert isinstance(d, sym.RangeIndex) and d.lower is None and d.upper is None, \
            f"Expected deferred dimension (:), got {d}"

    b_shape = routine.variable_map['b'].shape
    assert len(b_shape) == 3
    assert b_shape[0] == routine.variable_map['n']
    assert b_shape[1] == routine.variable_map['m']
    assert b_shape[2] == sym.Literal(10)

    # Check that deferred dimensions are kept in the declaration output
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    a_decl = next(d for d in decls if any(s.name.lower() == 'a' for s in d.symbols))
    b_decl = next(d for d in decls if any(s.name.lower() == 'b' for s in d.symbols))
    assert 'a(:,:,:)' in fgen(a_decl).lower().replace(' ', '')
    assert 'a(:,:,10)' not in fgen(a_decl).lower().replace(' ', '')
    assert 'b(n,m,10)' in fgen(b_decl).lower().replace(' ', '')


@pytest.mark.parametrize('frontend', available_frontends())
def test_get_locals_to_promote_skips_deferred_shape(frontend):
    """
    Test that :any:`PromoteLocalArrayTransformation.get_locals_to_promote`
    skips local arrays whose shape is entirely deferred (all ``:``
    dimensions), since the actual shape is not known at compile time for
    pointer/allocatable locals.

    Variables with explicit shapes should still be returned as candidates.
    """
    fcode = """
subroutine test_skip_deferred(ncol, nlev, istartcol, iendcol)
  implicit none
  integer, intent(in) :: ncol, nlev, istartcol, iendcol
  real :: normal_arr(nlev)
  real, pointer :: ptr_arr(:,:,:)
  real, allocatable :: alloc_arr(:,:)
  real, dimension(:) :: assumed_arr
  integer :: jlev, jcol

  normal_arr(:) = 0.0
  do jlev = 1, nlev
    normal_arr(jlev) = jlev
  end do
  ptr_arr(1, 1, 1) = 0.0
  alloc_arr(1, 1) = 0.0
  assumed_arr(1) = 0.0
end subroutine test_skip_deferred
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Wrap body in a vector section, as produced by SCCDevector
    body_nodes = list(routine.body.body)
    vector_section = Section(body=tuple(body_nodes), label='vector_section')
    routine.body = routine.body.clone(body=(vector_section,))

    horizontal = Dimension(
        name='horizontal', size=['ncol'], index='jcol',
        lower=['istartcol'], upper=['iendcol']
    )

    sections = [
        s for s in FindNodes(ir.Section).visit(routine.body)
        if s.label == 'vector_section'
    ]

    candidates = PromoteLocalArrayTransformation.get_locals_to_promote(
        routine, sections, horizontal
    )

    candidate_names = {c.name.lower() for c in candidates}

    # normal_arr should be a candidate (explicit shape, used in vector section)
    assert 'normal_arr' in candidate_names

    # Pointer, allocatable, and assumed-shape arrays should be skipped
    # (all have entirely deferred shapes)
    assert 'ptr_arr' not in candidate_names
    assert 'alloc_arr' not in candidate_names
    assert 'assumed_arr' not in candidate_names


@pytest.mark.parametrize('frontend', available_frontends())
def test_get_locals_to_promote_skips_constant_shape(frontend):
    """
    Test that :any:`PromoteLocalArrayTransformation.get_locals_to_promote`
    skips local arrays whose dimensions are all compile-time constants
    (integer literals or Fortran ``PARAMETER`` values).

    Only arrays with at least one variable (non-constant) dimension
    should be returned as promotion candidates.
    """
    fcode = """
subroutine test_skip_constant(ncol, nlev, istartcol, iendcol)
  implicit none
  integer, intent(in) :: ncol, nlev, istartcol, iendcol
  integer, parameter :: nband = 14
  real :: variable_arr(nlev)
  real :: literal_arr(10)
  real :: param_arr(nband)
  real :: mixed_arr(nband, nlev)
  integer :: jlev, jcol

  variable_arr(:) = 0.0
  literal_arr(:) = 0.0
  param_arr(:) = 0.0
  mixed_arr(:,:) = 0.0
  do jlev = 1, nlev
    variable_arr(jlev) = jlev
    mixed_arr(1, jlev) = jlev
  end do
end subroutine test_skip_constant
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Wrap body in a vector section
    body_nodes = list(routine.body.body)
    vector_section = Section(body=tuple(body_nodes), label='vector_section')
    routine.body = routine.body.clone(body=(vector_section,))

    horizontal = Dimension(
        name='horizontal', size=['ncol'], index='jcol',
        lower=['istartcol'], upper=['iendcol']
    )

    sections = [
        s for s in FindNodes(ir.Section).visit(routine.body)
        if s.label == 'vector_section'
    ]

    candidates = PromoteLocalArrayTransformation.get_locals_to_promote(
        routine, sections, horizontal
    )

    candidate_names = {c.name.lower() for c in candidates}

    # variable_arr has a variable dimension (nlev), so it should be promoted
    assert 'variable_arr' in candidate_names

    # mixed_arr has one constant and one variable dimension, so it should be promoted
    assert 'mixed_arr' in candidate_names

    # literal_arr has only a literal constant dimension, so it should not be promoted
    assert 'literal_arr' not in candidate_names

    # param_arr has only a PARAMETER constant dimension, so it should not be promoted
    assert 'param_arr' not in candidate_names
