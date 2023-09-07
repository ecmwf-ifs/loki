from pathlib import Path
import pytest

from conftest import jit_compile, clean_test, available_frontends
from loki import (
    OMNI, REGEX, Sourcefile, Subroutine, Import,
    FindNodes, FindInlineCalls, fgen, Assignment, IntLiteral, Module,
    SubroutineItem, Comment
)
from loki.transform import (
    Transformation, replace_selected_kind,  FileWriteTransformation
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='rename_transform')
def fixture_rename_transform():

    class RenameTransform(Transformation):
        """
        Simple `Transformation` object that renames subroutine and modules.
        """

        def transform_file(self, sourcefile, **kwargs):
            sourcefile.ir.prepend(
                Comment(text="! [Loki] RenameTransform applied")
            )

        def transform_subroutine(self, routine, **kwargs):
            routine.name += '_test'

        def transform_module(self, module, **kwargs):
            module.name += '_test'

    return RenameTransform()


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('method', ['source', 'transformation'])
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('recurse_to_contained_nodes', [True, False])
def test_transformation_apply(rename_transform, frontend, method, lazy, recurse_to_contained_nodes):
    """
    Apply a simple transformation that renames routines and modules, and
    test that this also works when the original source object was parsed
    using lazy construction.
    """
    fcode = """
module mymodule
  real(kind=4) :: myvar
end module mymodule

subroutine myroutine(a, b)
  real(kind=4), intent(inout) :: a, b

  a = a + b
end subroutine myroutine
"""
    # Let source apply transformation to all items and verify
    source = Sourcefile.from_source(fcode, frontend=REGEX if lazy else frontend)
    assert source._incomplete is lazy
    if method == 'source':
        if lazy:
            with pytest.raises(RuntimeError):
                source.apply(rename_transform)
            source.make_complete(frontend=frontend)
        source.apply(rename_transform, recurse_to_contained_nodes=recurse_to_contained_nodes)
    elif method == 'transformation':
        if lazy:
            with pytest.raises(RuntimeError):
                rename_transform.apply(source)
            source.make_complete(frontend=frontend)
        rename_transform.apply(source, recurse_to_contained_nodes=recurse_to_contained_nodes)
    else:
        raise ValueError(f'Unknown method "{method}"')
    assert not source._incomplete

    assert isinstance(source.ir.body[0], Comment)
    assert source.ir.body[0].text == '! [Loki] RenameTransform applied'

    # Without recursion, only source file object is changed
    if not recurse_to_contained_nodes:
        assert source.modules[0].name == 'mymodule'
        assert source.subroutines[0].name == 'myroutine'

        if method == 'source':
            source.modules[0].apply(rename_transform, recurse_to_contained_nodes=True)
            source.subroutines[0].apply(rename_transform, recurse_to_contained_nodes=True)
        else:
            rename_transform.apply(source.modules[0], recurse_to_contained_nodes=True)
            rename_transform.apply(source.subroutines[0], recurse_to_contained_nodes=True)

    assert source.modules[0].name == 'mymodule_test'
    assert source['mymodule_test'] == source.modules[0]
    assert source.subroutines[0].name == 'myroutine_test'
    assert source['myroutine_test'] == source.subroutines[0]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('target, apply_method', [
    ('module_routine', lambda transform, obj, **kwargs: obj.apply(transform, **kwargs)),
    ('myroutine', lambda transform, obj, **kwargs: transform.apply_subroutine(obj, **kwargs))
])
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('recurse_to_contained_nodes', [False, True])
def test_transformation_apply_subroutine(rename_transform, frontend, target, apply_method, lazy,
                                         recurse_to_contained_nodes):
    """
    Apply a simple transformation that renames routines and modules
    """
    fcode = """
module mymodule
  real(kind=4) :: myvar

contains

  subroutine module_routine(argument)
    real(kind=4), intent(inout) :: argument

    argument = member_func()

  contains
    function member_func() result(res)
      real(kind=4) :: res

      res = 4.
    end function member_func
  end subroutine module_routine
end module mymodule

subroutine myroutine(a, b)
  real(kind=4), intent(inout) :: a, b

  a = a + b
end subroutine myroutine
"""
    source = Sourcefile.from_source(fcode, frontend=REGEX if lazy else frontend)
    assert source._incomplete is lazy
    assert source[target]._incomplete is lazy

    if lazy:
        with pytest.raises(RuntimeError):
            apply_method(rename_transform, source[target])
        source[target].make_complete(frontend=frontend)
    apply_method(rename_transform, source[target], recurse_to_contained_nodes=recurse_to_contained_nodes)

    assert source._incomplete is lazy  # This should only have triggered a re-parse on the actual transformation target
    assert not source[f'{target}_test']._incomplete
    assert source.modules[0].name == 'mymodule'
    assert source['mymodule'] == source.modules[0]
    if target == 'module_routine':
        # Let only the inner module routine apply the transformation
        assert source.subroutines[0].name == 'myroutine'
        assert source['myroutine'] == source.subroutines[0]
    elif target == 'myroutine':
        # Apply transformation explicitly to the outer routine
        assert source.subroutines[0].name == 'myroutine_test'
        assert source['myroutine_test'] == source.subroutines[0]
    assert len(source.all_subroutines) == 2  # Ignore member func
    if target == 'module_routine':
        assert source.all_subroutines[1].name == 'module_routine_test'
        assert source['module_routine_test'] == source.all_subroutines[1]
        assert len(source['module_routine_test'].members) == 1
        if recurse_to_contained_nodes:
            assert source['module_routine_test'].members[0].name == 'member_func_test'
        else:
            assert source['module_routine_test'].members[0].name == 'member_func'
    elif target == 'myroutine':
        assert source.all_subroutines[1].name == 'module_routine'
        assert source['module_routine'] == source.all_subroutines[1]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('apply_method', [
    lambda transform, obj, **kwargs: obj.apply(transform, **kwargs),
    lambda transform, obj, **kwargs: transform.apply_module(obj, **kwargs)
])
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('recurse_to_contained_nodes', [False, True])
def test_transformation_apply_module(rename_transform, frontend, apply_method, lazy, recurse_to_contained_nodes):
    """
    Apply a simple transformation that renames routines and modules
    """
    fcode = """
module mymodule
  real(kind=4) :: myvar

contains

  subroutine module_routine(argument)
    real(kind=4), intent(inout) :: argument

    argument = argument  + 1.
  end subroutine module_routine
end module mymodule

subroutine myroutine(a, b)
  real(kind=4), intent(inout) :: a, b

  a = a + b
end subroutine myroutine
"""
    source = Sourcefile.from_source(fcode, frontend=REGEX if lazy else frontend)
    assert source._incomplete is lazy
    assert source['mymodule']._incomplete is lazy
    assert source['myroutine']._incomplete is lazy

    if lazy:
        with pytest.raises(RuntimeError):
            apply_method(rename_transform, source['mymodule'])
        source['mymodule'].make_complete(frontend=frontend)
    apply_method(rename_transform, source['mymodule'], recurse_to_contained_nodes=recurse_to_contained_nodes)

    assert source._incomplete is lazy
    assert not source['mymodule_test']._incomplete
    assert source['myroutine']._incomplete is lazy
    assert source.modules[0].name == 'mymodule_test'
    assert source['mymodule_test'] == source.modules[0]
    assert len(source.all_subroutines) == 2
    # Outer subroutine is untouched, since we apply all
    # transformations to anything in the module.
    assert source.subroutines[0].name == 'myroutine'
    assert source['myroutine'] == source.subroutines[0]

    if recurse_to_contained_nodes:
        assert source.all_subroutines[1].name == 'module_routine_test'
        assert source['module_routine_test'] == source.all_subroutines[1]
    else:
        assert source.all_subroutines[1].name == 'module_routine'
        assert source['module_routine'] == source.all_subroutines[1]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_replace_selected_kind(here, frontend):
    """
    Test correct replacement of all `selected_x_kind` calls by
    iso_fortran_env constant.
    """
    fcode = """
subroutine transform_replace_selected_kind(i, a)
  use iso_fortran_env, only: int8
  implicit none
  integer, parameter :: jprm = selected_real_kind(6,37)
  integer(kind=selected_int_kind(9)), intent(out) :: i
  real(kind=selected_real_kind(13,300)), intent(out) :: a
  integer(kind=int8) :: j = 1
  integer(kind=selected_int_kind(1)) :: k = 9
  real(kind=selected_real_kind(7)) :: b = 5._jprm
  real(kind=selected_real_kind(r=2, p=4)) :: c = 1.

  i = j + k
  a = b + c + real(4, kind=selected_real_kind(6, r=37))
end subroutine transform_replace_selected_kind
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    imports = FindNodes(Import).visit(routine.spec)
    assert len(imports) == 1 and imports[0].module.lower() == 'iso_fortran_env'
    assert len(imports[0].symbols) == 1 and imports[0].symbols[0].name.lower() == 'int8'

    # Test the original implementation
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    i, a = function()
    assert i == 10
    assert a == 10.

    # Apply transformation and check imports
    replace_selected_kind(routine)
    assert not [call for call in FindInlineCalls().visit(routine.ir)
                if call.name.lower().startswith('selected')]

    imports = FindNodes(Import).visit(routine.spec)
    assert len(imports) == 1 and imports[0].module.lower() == 'iso_fortran_env'

    source = fgen(routine).lower()
    assert not 'selected_real_kind' in source
    assert not 'selected_int_kind' in source

    if frontend == OMNI:
        # F£$%^% OMNI replaces randomly SOME selected_real_kind calls by
        # (wrong!) integer kinds
        symbols = {'int8', 'real32', 'real64'}
    else:
        symbols = {'int8', 'int32', 'real32', 'real64'}

    assert len(imports[0].symbols) == len(symbols)
    assert {s.name.lower() for s in imports[0].symbols} == symbols

    # Test the transformed implementation
    iso_filepath = here/(f'{routine.name}_replaced_{frontend}.f90')
    iso_function = jit_compile(routine, filepath=iso_filepath, objname=routine.name)

    i, a = iso_function()
    assert i == 10
    assert a == 10.

    clean_test(filepath)
    clean_test(iso_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('post_apply_rescope_symbols', [True, False])
def test_transformation_post_apply_subroutine(here, frontend, post_apply_rescope_symbols):
    """Verify that post_apply is called for subroutines."""

    #### Test that rescoping is applied and effective ####

    tmp_routine = Subroutine('some_routine')
    class ScopingErrorTransformation(Transformation):
        """Intentionally idiotic transformation that introduces a scoping error."""

        def transform_subroutine(self, routine, **kwargs):
            i = routine.variable_map['i']
            j = i.clone(name='j', scope=tmp_routine, type=i.type.clone(intent=None))
            routine.variables += (j,)
            routine.body.append(Assignment(lhs=j, rhs=IntLiteral(2)))
            routine.body.append(Assignment(lhs=i, rhs=j))
            routine.name += '_transformed'
            assert routine.variable_map['j'].scope is tmp_routine

    fcode = """
subroutine transformation_post_apply(i)
  integer, intent(out) :: i
  i = 1
end subroutine transformation_post_apply
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Test the original implementation
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    i = function()
    assert i == 1

    # Apply transformation and make sure variable scope is correct
    routine.apply(ScopingErrorTransformation(), post_apply_rescope_symbols=post_apply_rescope_symbols)
    if post_apply_rescope_symbols:
        # Scope is correct
        assert routine.variable_map['j'].scope is routine
    else:
        # Scope is wrong
        assert routine.variable_map['j'].scope is tmp_routine

    new_filepath = here/(f'{routine.name}_{frontend}.f90')
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    i = new_function()
    assert i == 2

    clean_test(filepath)
    clean_test(new_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('post_apply_rescope_symbols', [True, False])
def test_transformation_post_apply_module(here, frontend, post_apply_rescope_symbols):
    """Verify that post_apply is called for modules."""

    #### Test that rescoping is applied and effective ####

    tmp_scope = Module('some_module')
    class ScopingErrorTransformation(Transformation):
        """Intentionally idiotic transformation that introduces a scoping error."""

        def transform_module(self, module, **kwargs):
            i = module.variable_map['i']
            j = i.clone(name='j', scope=tmp_scope, type=i.type.clone(intent=None))
            module.variables += (j,)
            routine = module.subroutines[0]
            routine.body.prepend(Assignment(lhs=i, rhs=j))
            routine.body.prepend(Assignment(lhs=j, rhs=IntLiteral(2)))
            module.name += '_transformed'
            assert module.variable_map['j'].scope is tmp_scope

    fcode = """
module transformation_module_post_apply
  integer :: i = 0
contains
  subroutine test_post_apply(ret)
    integer, intent(out) :: ret
    i = i + 1
    ret = i
  end subroutine test_post_apply
end module transformation_module_post_apply
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    # Test the original implementation
    filepath = here/(f'{module.name}_{frontend}_{post_apply_rescope_symbols!s}.f90')
    mod = jit_compile(module, filepath=filepath, objname=module.name)

    i = mod.test_post_apply()
    assert i == 1

    # Apply transformation
    module.apply(ScopingErrorTransformation(), post_apply_rescope_symbols=post_apply_rescope_symbols)
    if post_apply_rescope_symbols:
        # Scope is correct
        assert module.variable_map['j'].scope is module
    else:
        # Scope is wrong
        assert module.variable_map['j'].scope is tmp_scope

    new_filepath = here/(f'{module.name}_{frontend}_{post_apply_rescope_symbols!s}.f90')
    new_mod = jit_compile(module, filepath=new_filepath, objname=module.name)

    i = new_mod.test_post_apply()
    assert i == 3

    clean_test(filepath)
    clean_test(new_filepath)


def test_transformation_file_write(here):
    """Verify that files get written with correct filenames"""

    fcode = """
subroutine rick()
  print *, "PRINT ME!"
end subroutine rick
"""
    source = Sourcefile.from_source(fcode)
    source.path = Path('rick.F90')
    item = SubroutineItem(name='#rick', source=source)

    # Test default file writes
    ricks_path = here/'rick.loki.F90'
    if ricks_path.exists():
        ricks_path.unlink()
    FileWriteTransformation(builddir=here).apply(source=source, item=item)
    assert ricks_path.exists()
    ricks_path.unlink()

    # Test mode and suffix overrides
    ricks_path = here/'rick.roll.java'
    if ricks_path.exists():
        ricks_path.unlink()
    FileWriteTransformation(builddir=here, mode='roll', suffix='.java').apply(source=source, item=item)
    assert ricks_path.exists()
    ricks_path.unlink()
