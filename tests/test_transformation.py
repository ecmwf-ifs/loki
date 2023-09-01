from pathlib import Path
import pytest

from conftest import jit_compile, clean_test, available_frontends
from loki import (
    OMNI, REGEX, OFP, Sourcefile, Subroutine, CallStatement, Import,
    FindNodes, FindInlineCalls, fgen, Assignment, IntLiteral, Module,
    SubroutineItem, Intrinsic, Comment
)
from loki.transform import (
    Transformation, DependencyTransformation, replace_selected_kind,
    FileWriteTransformation
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
def test_dependency_transformation_module_imports(frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines via module imports.
    """

    kernel = Sourcefile.from_source(source="""
MODULE kernel_mod
CONTAINS
    SUBROUTINE kernel(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    a = 1
    b = 2
    c = 3
  END SUBROUTINE kernel
END MODULE kernel_mod
""", frontend=frontend)

    driver = Sourcefile.from_source(source="""
MODULE driver_mod
  USE kernel_mod, only: kernel
CONTAINS
  SUBROUTINE driver(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
  END SUBROUTINE driver
END MODULE driver_mod
""", frontend=frontend)

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')
    # Because the renaming is intended to be applied to the routines as well as the enclosing module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver.apply(transformation, role='driver', targets='kernel', recurse_to_contained_nodes=True)

    # Check that the basic entity names in the kernel source have changed
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the entity names in the driver have not changed
    assert driver.all_subroutines[0].name == 'driver'
    assert driver['driver'] == driver.all_subroutines[0]
    assert driver.modules[0].name == 'driver_mod'
    assert driver['driver_mod'] == driver.modules[0]

    # Check that calls and imports have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver_mod'].spec)
    assert len(imports) == 1
    assert isinstance(imports[0], Import)
    assert driver['driver_mod'].spec.body[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in driver['driver_mod'].spec.body[0].symbols]


@pytest.mark.parametrize('frontend', available_frontends())
def test_dependency_transformation_globalvar_imports(frontend):
    """
    Test that global variable imports are not renamed as a
    call statement would be.
    """

    kernel = Sourcefile.from_source(source="""
MODULE kernel_mod
    INTEGER :: some_const
CONTAINS
    SUBROUTINE kernel(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    a = 1
    b = 2
    c = 3
  END SUBROUTINE kernel
END MODULE kernel_mod
""", frontend=frontend)

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
    USE kernel_mod, only: kernel
    USE kernel_mod, only: some_const
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')
    # Because the renaming is intended to be applied to the routines as well as the enclosing module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver['driver'].apply(transformation, role='driver', targets=('kernel', 'some_const'))

    # Check that the global variable declaration remains unchanged
    assert kernel.modules[0].variables[0].name == 'some_const'

    # Check that calls and matching import have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 2
    assert isinstance(imports[0], Import)
    assert driver['driver'].spec.body[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in driver['driver'].spec.body[0].symbols]

    # Check that global variable import remains unchanged
    assert isinstance(imports[1], Import)
    assert driver['driver'].spec.body[1].module == 'kernel_mod'
    assert 'some_const' in [str(s) for s in driver['driver'].spec.body[1].symbols]


@pytest.mark.parametrize('frontend', available_frontends())
def test_dependency_transformation_globalvar_imports_driver_mod(frontend):
    """
    Test that global variable imports are not renamed as a
    call statement would be.
    """

    kernel = Sourcefile.from_source(source="""
MODULE kernel_mod
    INTEGER :: some_const
CONTAINS
    SUBROUTINE kernel(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    a = 1
    b = 2
    c = 3
  END SUBROUTINE kernel
END MODULE kernel_mod
""", frontend=frontend)

    driver = Sourcefile.from_source(source="""
MODULE DRIVER_MOD
    USE kernel_mod, only: kernel
    USE kernel_mod, only: some_const
CONTAINS
SUBROUTINE driver(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
END SUBROUTINE driver
END MODULE DRIVER_MOD
""", frontend=frontend)

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')
    # Because the renaming is intended to be applied to the routines as well as the enclosing module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver.apply(transformation, role='driver', targets=('kernel', 'some_const'), recurse_to_contained_nodes=True)

    # Check that the global variable declaration remains unchanged
    assert kernel.modules[0].variables[0].name == 'some_const'

    # Check that calls and matching import have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver_mod'].spec)
    assert len(imports) == 2
    assert isinstance(imports[0], Import)
    assert driver['driver_mod'].spec.body[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in driver['driver_mod'].spec.body[0].symbols]

    # Check that global variable import remains unchanged
    assert isinstance(imports[1], Import)
    assert driver['driver_mod'].spec.body[1].module == 'kernel_mod'
    assert 'some_const' in [str(s) for s in driver['driver_mod'].spec.body[1].symbols]

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'C-imports need pre-processing for OMNI')]))
def test_dependency_transformation_header_includes(here, frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines via c-header includes.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

#include "kernel.intfb.h"

  CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
""", frontend=frontend)

    # Ensure header file does not exist a-priori
    header_file = here/'kernel_test.intfb.h'
    if header_file.exists():
        header_file.unlink()

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='strict', include_path=here)
    kernel['kernel'].apply(transformation, role='kernel')
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the subroutine name in the kernel source has changed
    assert len(kernel.modules) == 0
    assert len(kernel.subroutines) == 1
    assert kernel.subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]

    # Check that the driver name has not changed
    assert len(kernel.modules) == 0
    assert len(kernel.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that the import has been updated
    assert '#include "kernel.intfb.h"' not in driver.to_fortran()
    assert '#include "kernel_test.intfb.h"' in driver.to_fortran()

    # Check that header file was generated and clean up
    assert header_file.exists()
    header_file.unlink()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'C-imports need pre-processing for OMNI')]))
def test_dependency_transformation_module_wrap(frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines automatic module wrapping of the kernel.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

#include "kernel.intfb.h"

  CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    assert imports[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in imports[0].symbols]


@pytest.mark.parametrize('frontend', available_frontends())
def test_dependency_transformation_replace_interface(frontend):
    """
    Test injection of suffixed kernels defined in interface block
    into unchanged driver routines automatic module wrapping of the kernel.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  IMPLICIT NONE
  INTERFACE
    SUBROUTINE KERNEL(a, b, c)
      INTEGER, INTENT(INOUT) :: a, b, c
    END SUBROUTINE KERNEL
  END INTERFACE

  INTEGER, INTENT(INOUT) :: a, b, c

  CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    if frontend == OMNI:
        assert imports[0].module == 'kernel_test_mod'
        assert 'kernel_test' in [str(s) for s in imports[0].symbols]
    else:
        assert imports[0].module == 'KERNEL_test_mod'
        assert 'KERNEL_test' in [str(s) for s in imports[0].symbols]

    # Check that the newly generated USE statement appears before IMPLICIT NONE
    nodes = FindNodes((Intrinsic, Import)).visit(driver['driver'].spec)
    assert len(nodes) == 2
    assert isinstance(nodes[1], Intrinsic)
    assert nodes[1].text.lower() == 'implicit none'


@pytest.mark.parametrize('frontend', available_frontends(
                         xfail=[(OFP, 'OFP does not correctly handle result variable declaration.')]))
def test_dependency_transformation_inline_call(frontend):
    """
    Test injection of suffixed kernel, accessed through inline function call.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTERFACE
    INTEGER FUNCTION kernel(a)
      INTEGER, INTENT(IN) :: a
    END FUNCTION kernel
  END INTERFACE

  INTEGER, INTENT(INOUT) :: a, b, c

  a = kernel(a)
  b = kernel(a)
  c = kernel(c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
INTEGER FUNCTION kernel(a)
  INTEGER, INTENT(IN) :: a

  kernel = 2*a
END FUNCTION kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel['kernel_test'].is_function
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the return name has been added as a variable
    assert 'kernel_test' in kernel['kernel_test'].variables

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = tuple(FindInlineCalls().visit(driver['driver'].body))
    assert len(calls) == 2
    calls = tuple(FindInlineCalls(unique=False).visit(driver['driver'].body))
    assert len(calls) == 3
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    assert imports[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in imports[0].symbols]


@pytest.mark.parametrize('frontend', available_frontends(
                         xfail=[(OFP, 'OFP does not correctly handle result variable declaration.')]))
def test_dependency_transformation_inline_call_result_var(frontend):
    """
    Test injection of suffixed kernel, accessed through inline function call.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTERFACE
    FUNCTION kernel(a) RESULT(ret)
      INTEGER, INTENT(IN) :: a
      INTEGER :: ret
    END FUNCTION kernel
  END INTERFACE

  INTEGER, INTENT(INOUT) :: a, b, c

  a = kernel(a)
  b = kernel(a)
  c = kernel(c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
FUNCTION kernel(a) RESULT(ret)
  INTEGER, INTENT(IN) :: a
  INTEGER :: ret

  ret = 2*a
END FUNCTION kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel['kernel_test'].is_function
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = tuple(FindInlineCalls().visit(driver['driver'].body))
    assert len(calls) == 2
    calls = tuple(FindInlineCalls(unique=False).visit(driver['driver'].body))
    assert len(calls) == 3
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    assert imports[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in imports[0].symbols]


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
