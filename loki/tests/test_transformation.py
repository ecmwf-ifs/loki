from functools import partial
from pathlib import Path
import pytest

from conftest import jit_compile, clean_test
from loki import (
    Sourcefile, Subroutine, Import, FindNodes, FindInlineCalls, fgen,
    Assignment, IntLiteral, Module, ProcedureItem, Comment
)
from loki.frontend import available_frontends, OMNI, REGEX
from loki.transform import (
    Transformation, replace_selected_kind, FileWriteTransformation, Pipeline
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
def test_transformation_apply(rename_transform, frontend, method, lazy):
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
        source.apply(rename_transform)
    elif method == 'transformation':
        if lazy:
            with pytest.raises(RuntimeError):
                rename_transform.apply(source)
            source.make_complete(frontend=frontend)
        rename_transform.apply(source)
    else:
        raise ValueError(f'Unknown method "{method}"')
    assert not source._incomplete

    assert isinstance(source.ir.body[0], Comment)
    assert source.ir.body[0].text == '! [Loki] RenameTransform applied'

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
def test_transformation_apply_subroutine(rename_transform, frontend, target, apply_method, lazy):
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
    apply_method(rename_transform, source[target])

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
def test_transformation_apply_module(rename_transform, frontend, apply_method, lazy):
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
    apply_method(rename_transform, source['mymodule'])

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
    item = ProcedureItem(name='#rick', source=source)

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

    # Test writing with "items" only (as in file graph traversal)
    ricks_path = here/'rick.loki.F90'
    if ricks_path.exists():
        ricks_path.unlink()
    FileWriteTransformation(builddir=here).apply(source=source, items=(item,))
    assert ricks_path.exists()
    ricks_path.unlink()

    # Check error behaviour if no item provided
    with pytest.raises(ValueError):
        FileWriteTransformation(builddir=here).apply(source=source)


def test_transformation_pipeline_simple():
    """
    Test the instantiation of a :any:`Pipeline` from a partial definition.
    """

    class PrependTrafo(Transformation):
        def __init__(self, name='Rick', relaxed=False):
            self.name = name
            self.relaxed = relaxed

        def transform_subroutine(self, routine, **kwargs):
            greeting = 'Whazzup' if self.relaxed else 'Hello'
            routine.body.prepend(Comment(text=f'! {greeting} {self.name}'))

    class AppendTrafo(Transformation):
        def __init__(self, name='Dave', in_french=False):
            self.name = name
            self.in_french = in_french

        def transform_subroutine(self, routine, **kwargs):
            greeting = 'Au revoir' if self.in_french else 'Goodbye'
            routine.body.append(Comment(text=f'! {greeting}, {self.name}'))

    # Define a pipline as a combination of transformation classes
    # and a set pre-defined constructor flags
    GreetingPipeline = partial(
        Pipeline, classes=(PrependTrafo, AppendTrafo), relaxed=True
    )

    # Instantiate the pipeline object with additional constructor flags
    pipeline = GreetingPipeline(name='Bob', in_french=True)

    assert pipeline.transformations and len(pipeline.transformations) == 2
    assert isinstance(pipeline.transformations[0], PrependTrafo)
    assert pipeline.transformations[0].name == 'Bob'
    assert isinstance(pipeline.transformations[1], AppendTrafo)
    assert pipeline.transformations[1].name == 'Bob'
    assert pipeline.transformations[1].in_french

    # Now apply the pipeline to a simple subroutine
    fcode = """
subroutine test_pipeline
  integer :: i
  real :: a, b

  do i=1,3
    a = a + b
  end do
end subroutine test_pipeline
"""
    routine = Subroutine.from_source(fcode)
    pipeline.apply(routine)

    assert isinstance(routine.body.body[0], Comment)
    assert routine.body.body[0].text == '! Whazzup Bob'
    assert isinstance(routine.body.body[-1], Comment)
    assert routine.body.body[-1].text == '! Au revoir, Bob'


def test_transformation_pipeline_constructor():
    """
    Test the correct argument handling when instantiating a
    :any:`Pipeline` from a partial definitions.
    """

    class DoSomethingTrafo(Transformation):
        def __init__(self, a, b=None, c=True, d='yes'):
            self.a = a
            self.b = b
            self.c = c
            self.d = d

    class DoSomethingElseTrafo(Transformation):
        def __init__(self, b=None, d='no'):
            self.b = b
            self.d = d

    MyPipeline = partial(
        Pipeline, classes=(
            DoSomethingTrafo,
            DoSomethingElseTrafo,
        ),
        a=42
    )

    p1 = MyPipeline(b=66, d='yes')
    assert p1.transformations[0].a == 42
    assert p1.transformations[0].b == 66
    assert p1.transformations[0].c is True
    assert p1.transformations[0].d == 'yes'
    assert p1.transformations[1].b == 66
    assert p1.transformations[1].d == 'yes'

    # Now we use inheritance to propagate defaults

    class DoSomethingDifferentTrafo(DoSomethingTrafo):
        def __init__(self, e=1969, **kwargs):
            super().__init__(**kwargs)
            self.e = e

    MyOtherPipeline = partial(
        Pipeline, classes=(
            DoSomethingDifferentTrafo,
            DoSomethingElseTrafo,
        ),
        a=42
    )

    # Now check if inheritance works
    p2 = MyOtherPipeline(b=66, d='yes', e=1977)
    assert p2.transformations[0].a == 42
    assert p2.transformations[0].b == 66
    assert p2.transformations[0].c is True
    assert p2.transformations[0].d == 'yes'
    assert p2.transformations[0].e == 1977
    assert p2.transformations[1].b == 66
    assert p2.transformations[1].d == 'yes'


def test_transformation_pipeline_compose():
    """
    Test append / prepend functionalities of :any:`Pipeline` objects.
    """

    fcode = """
subroutine test_pipeline_compose(a)
  implicit none
  real, intent(inout) :: a
  a = a + 1.0
end subroutine test_pipeline_compose
"""

    class YesTrafo(Transformation):
        def transform_subroutine(self, routine, **kwargs):
            routine.body.append( Comment(text='! Yes !') )

    class NoTrafo(Transformation):
        def transform_subroutine(self, routine, **kwargs):
            routine.body.append( Comment(text='! No !') )

    class MaybeTrafo(Transformation):
        def transform_subroutine(self, routine, **kwargs):
            routine.body.append( Comment(text='! Maybe !') )

    class MaybeNotTrafo(Transformation):
        def transform_subroutine(self, routine, **kwargs):
            routine.body.append( Comment(text='! Maybe not !') )

    pipeline = Pipeline(classes=(YesTrafo, NoTrafo))
    pipeline.prepend(MaybeTrafo())
    pipeline.append(MaybeNotTrafo())

    routine = Subroutine.from_source(fcode)
    pipeline.apply(routine)

    comments = FindNodes(Comment).visit(routine.body)
    assert len(comments) == 4
    assert comments[0].text == '! Maybe !'
    assert comments[1].text == '! Yes !'
    assert comments[2].text == '! No !'
    assert comments[3].text == '! Maybe not !'

    # Now try the same trick, but with the native addition API
    pipe_a = Pipeline(classes=(MaybeTrafo,))
    pipe_b = Pipeline(classes=(MaybeNotTrafo,YesTrafo))
    pipe = YesTrafo() + pipe_a + pipe_b + NoTrafo()

    with pytest.raises(TypeError):
        pipe += lambda t: t

    routine = Subroutine.from_source(fcode)
    pipe.apply(routine)

    comments = FindNodes(Comment).visit(routine.body)
    assert len(comments) == 5
    assert comments[0].text == '! Yes !'
    assert comments[1].text == '! Maybe !'
    assert comments[2].text == '! Maybe not !'
    assert comments[3].text == '! Yes !'
    assert comments[4].text == '! No !'

    # Check that the string representation is sane
    assert '<YesTrafo  [test_transformation]' in str(pipe)
    assert '<MaybeTrafo  [test_transformation]' in str(pipe)
    assert '<MaybeNotTrafo  [test_transformation]' in str(pipe)
    assert '<YesTrafo  [test_transformation]' in str(pipe)
    assert '<NoTrafo  [test_transformation]' in str(pipe)
