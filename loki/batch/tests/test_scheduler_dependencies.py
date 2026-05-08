# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

from xml.etree.ElementTree import ParseError

import pytest

from loki import (
    Array, BasicType, CaseInsensitiveDict, DerivedType, ProcedureType,
    Scalar, Sourcefile, Subroutine, available_frontends, fexprgen,
    graphviz_present
)
from loki.batch import (
    ExternalItem, InterfaceItem, ModuleItem, ProcedureItem, ProcessingStrategy,
    Scheduler, SchedulerConfig, Transformation, TypeDefItem
)
from loki.expression import ProcedureSymbol
from loki.frontend import FP, HAVE_FP, OMNI, REGEX
from loki.ir import FindInlineCalls, FindNodes, FindVariables, nodes as ir

from .conftest import VisGraphWrapper

pytestmark = pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')


@pytest.fixture(name='proj_typebound_dependencies')
def fixture_proj_typebound_dependencies():
    return {
        '#driver': (
            'typebound_item',
            'typebound_item#some_type%other_routine',
            'typebound_item#some_type%some_routine',
            'typebound_header',
            'typebound_header#header_type%member_routine',
            'typebound_header#header_type%routine',
            'typebound_other#other_type',
            'typebound_other#other_type%member',
            'typebound_other#other_type%var%member_routine',
        ),
        'typebound_item': ('typebound_header',),
        'typebound_item#some_type': (),
        'typebound_item#some_type%some_routine': ('typebound_item#some_routine',),
        'typebound_item#some_type%other_routine': ('typebound_item#other_routine',),
        'typebound_item#some_type%routine': ('typebound_item#module_routine',),
        'typebound_item#some_type%routine1': ('typebound_item#routine1',),
        'typebound_item#some_type%routine2': ('typebound_item#routine',),
        'typebound_item#routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%some_routine',
        ),
        'typebound_item#routine1': (
            'typebound_item#module_routine',
            'typebound_item#some_type'
        ),
        'typebound_item#some_routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%routine',
        ),
        'typebound_item#other_routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%routine1',
            'typebound_item#some_type%routine2',
            'typebound_header#abor1'
        ),
        'typebound_item#module_routine': (),
        'typebound_header': (),
        'typebound_header#header_type': (),
        'typebound_header#header_type%member_routine': ('typebound_header#header_member_routine',),
        'typebound_header#header_member_routine': ('typebound_header#header_type',),
        'typebound_header#header_type%routine': (
            'typebound_header#header_type%routine_real',
            'typebound_header#header_type%routine_integer'
        ),
        'typebound_header#header_type%routine_real': ('typebound_header#header_routine_real',),
        'typebound_header#header_routine_real': ('typebound_header#header_type',),
        'typebound_header#header_type%routine_integer': ('typebound_header#routine_integer',),
        'typebound_header#routine_integer': ('typebound_header#header_type',),
        'typebound_header#abor1': (),
        'typebound_other#other_type': ('typebound_header#header_type',),
        'typebound_other#other_type%member': ('typebound_other#other_member',),
        'typebound_other#other_member': (
            'typebound_header#header_member_routine',
            'typebound_other#other_type',
            'typebound_other#other_type%var%member_routine'
        ),
        'typebound_other#other_type%var%member_routine': ('typebound_header#header_type%member_routine',)
    }


@pytest.fixture(name='loki_69_dir')
def fixture_loki_69_dir(testdir):
    """
    Fixture to write test file for LOKI-69 test.
    """
    fcode = """
subroutine random_call_0(v_out,v_in,v_inout)
implicit none

    real(kind=jprb),intent(in)  :: v_in
    real(kind=jprb),intent(out)  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout


end subroutine random_call_0

!subroutine random_call_1(v_out,v_in,v_inout)
!implicit none
!
!  real(kind=jprb),intent(in)  :: v_in
!  real(kind=jprb),intent(out)  :: v_out
!  real(kind=jprb),intent(inout)  :: v_inout
!
!
!end subroutine random_call_1

subroutine random_call_2(v_out,v_in,v_inout)
implicit none

    real(kind=jprb),intent(in)  :: v_in
    real(kind=jprb),intent(out)  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout


end subroutine random_call_2

subroutine test(v_out,v_in,v_inout,some_logical)
implicit none

    real(kind=jprb),intent(in   )  :: v_in
    real(kind=jprb),intent(out  )  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout

    logical,intent(in)             :: some_logical

    v_inout = 0._jprb
    if(some_logical)then
        call random_call_0(v_out,v_in,v_inout)
    endif

    if(some_logical) call random_call_2

end subroutine test
    """.strip()

    dirname = testdir/'loki69'
    dirname.mkdir(exist_ok=True)
    filename = dirname/'test.F90'
    filename.write_text(fcode)
    yield dirname
    try:
        filename.unlink()
        dirname.rmdir()
    except FileNotFoundError:
        pass


def test_scheduler_enrichment(testdir, config, frontend, tmp_path):
    projA = testdir/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=['driverA'], frontend=frontend, xmods=[tmp_path]
    )

    for item in scheduler.sgraph:
        if not isinstance(item, ProcedureItem):
            continue
        dependency_map = CaseInsensitiveDict(
            (item_.local_name, item_) for item_ in scheduler.sgraph.successors(item)
        )
        for call in FindNodes(ir.CallStatement).visit(item.ir.body):
            if call_item := dependency_map.get(str(call.name)):
                assert call.routine is call_item.ir


@pytest.mark.parametrize('seed', ['driverA', 'driverA_mod#driverA'])
def test_scheduler_definitions(testdir, config, frontend, seed, tmp_path):
    """
    Create a simple task graph and inject type info via `definitions`.
    """
    projA = testdir/'sources/projA'

    header = Sourcefile.from_file(projA/'module/header_mod.f90', frontend=frontend)

    scheduler = Scheduler(
        paths=projA, definitions=header['header_mod'], includes=projA/'include',
        config=config, seed_routines=[seed], frontend=frontend, xmods=[tmp_path]
    )

    driver = scheduler.item_factory.item_cache['drivera_mod#drivera'].ir
    call = FindNodes(ir.CallStatement).visit(driver.body)[0]
    assert call.arguments[0].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[0].shape) == '(:,)'
    assert call.arguments[1].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[1].shape) == '(3, 3)'


def test_scheduler_module_dependency(testdir, config, frontend, tmp_path):
    """
    Ensure dependency chasing is done correctly, even with surboutines
    that do not match module names.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 | --> routine_one -> routine_two
    """
    projA = testdir/'sources/projA'
    projC = testdir/'sources/projC'

    scheduler = Scheduler(
        paths=[projA, projC], includes=projA/'include', config=config,
        seed_routines=['driverC_mod#driverC'], frontend=frontend, xmods=[tmp_path]
    )

    expected_dependencies = {
        'driverc_mod#driverc': ('header_mod', 'header_mod#header_type', 'kernelc_mod#kernelc',),
        'kernelc_mod#kernelc': ('compute_l1_mod#compute_l1', 'proj_c_util_mod#routine_one',),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'proj_c_util_mod#routine_one': ('proj_c_util_mod#routine_two',),
        'proj_c_util_mod#routine_two': (),
        'header_mod#header_type': (),
        'header_mod': (),
    }
    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # Ensure that we got the right routines from the module
    assert scheduler['proj_c_util_mod#routine_one'].ir.name == 'routine_one'
    assert scheduler['proj_c_util_mod#routine_two'].ir.name == 'routine_two'


def test_scheduler_module_dependencies_unqualified(testdir, config, frontend, tmp_path):
    """
    Ensure dependency chasing is done correctly for unqualified module imports.

    projA: driverD -> kernelD -> compute_l1<replicated> -> compute_l2
                           |
                    < proj_c_util_mod>
                           |
    projC:                 | --> routine_one -> routine_two
    """
    projA = testdir/'sources/projA'
    projC = testdir/'sources/projC'

    scheduler = Scheduler(
        paths=[projA, projC], includes=projA/'include', config=config,
        seed_routines=['driverD_mod#driverD'], frontend=frontend, xmods=[tmp_path]
    )

    expected_dependencies = {
        'driverd_mod#driverd': ('kerneld_mod#kerneld', 'header_mod', 'header_mod#header_type'),
        'kerneld_mod#kerneld': ('compute_l1_mod#compute_l1', 'proj_c_util_mod#routine_one'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'proj_c_util_mod#routine_one': ('proj_c_util_mod#routine_two',),
        'proj_c_util_mod#routine_two': (),
        'header_mod#header_type': (),
        'header_mod': (),
    }
    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # Ensure that we got the right routines from the module
    assert scheduler['proj_c_util_mod#routine_one'].ir.name == 'routine_one'
    assert scheduler['proj_c_util_mod#routine_two'].ir.name == 'routine_two'


@pytest.mark.parametrize('strict', [True, False])
def test_scheduler_missing_files(testdir, config, frontend, strict, tmp_path):
    """
    Ensure that ``strict=True`` triggers failure if source paths are
    missing and that ``strict=False`` goes through gracefully.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 < cannot find path >
    """
    projA = testdir/'sources/projA'

    config['default']['strict'] = strict
    scheduler = Scheduler(
        paths=[projA], includes=projA/'include', config=config,
        seed_routines=['driverC_mod#driverC'], frontend=frontend, xmods=[tmp_path]
    )

    expected_dependencies = {
        'driverc_mod#driverc': ('kernelc_mod#kernelc', 'header_mod#header_type', 'header_mod'),
        'kernelc_mod#kernelc': ('compute_l1_mod#compute_l1', 'proj_c_util_mod#routine_one'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'header_mod#header_type': (),
        'header_mod': (),
        'proj_c_util_mod#routine_one': (),
    }
    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    assert isinstance(scheduler['proj_c_util_mod#routine_one'], ExternalItem)
    # Ensure that the missing items are not in the graph
    assert 'proj_c_util_mod#routine_two' not in scheduler.items

    class CheckApply(Transformation):

        def apply(self, source, post_apply_rescope_symbols=False, plan_mode=False, **kwargs):
            assert 'item' in kwargs
            assert not isinstance(kwargs['item'], ExternalItem)
            super().apply(
                source, post_apply_rescope_symbols=post_apply_rescope_symbols,
                plan_mode=plan_mode, **kwargs
            )

    # Check processing with missing items
    if strict:
        with pytest.raises(RuntimeError):
            scheduler.process(CheckApply())
    else:
        scheduler.process(CheckApply())


def test_scheduler_item_dependencies(testdir, tmp_path):
    """
    Make sure children are correct and unique for items
    """
    config = SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routines': {
            'driver': {'role': 'driver'},
            'another_driver': {'role': 'driver'}
        }
    })

    proj_multi_mode = testdir/'sources/projHoist'

    scheduler = Scheduler(paths=proj_multi_mode, config=config, xmods=[tmp_path])

    assert tuple(
        call.name for call in scheduler['transformation_module_hoist#driver'].dependencies
    ) == ('kernel1', 'kernel2')
    assert tuple(
        call.name for call in scheduler['transformation_module_hoist#another_driver'].dependencies
    ) == ('kernel1',)
    assert not scheduler['subroutines_mod#kernel1'].dependencies
    assert tuple(
        call.name for call in scheduler['subroutines_mod#kernel2'].dependencies
    ) == ('device1', 'device2')
    assert tuple(
        call.name for call in scheduler['subroutines_mod#device1'].dependencies
    ) == ('device2',)
    assert not scheduler['subroutines_mod#device2'].dependencies


def test_scheduler_loki_69(loki_69_dir, tmp_path):
    """
    Test compliance of scheduler with edge cases reported in LOKI-69.
    """
    config = {
        'default': {
            'expand': True,
            'strict': True,
        },
    }

    scheduler = Scheduler(paths=loki_69_dir, seed_routines=['test'], config=config, xmods=[tmp_path])

    assert sorted(scheduler.item_factory.item_cache.keys()) == [
        '#random_call_0', '#random_call_2', '#test',
        str(loki_69_dir/'test.f90').lower()
    ]
    assert '#random_call_1' not in scheduler.item_factory

    children_map = {
        '#test': ('#random_call_0', '#random_call_2'),
        '#random_call_0': (),
        '#random_call_2': ()
    }
    assert len(scheduler.items) == len(children_map)
    for item in scheduler.items:
        assert set(scheduler.sgraph.successors(item)) == set(children_map[item.name])


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_scopes(tmp_path, testdir, config, frontend):
    """
    Test discovery with import renames and duplicate names in separate scopes

    driver ----> kernel1_mod#kernel ----> kernel1_impl#kernel_impl
      |
      +--------> kernel2_mod#kernel ----> kernel2_impl#kernel_impl
    """
    proj = testdir/'sources/projScopes'

    scheduler = Scheduler(paths=proj, seed_routines=['driver'], config=config, frontend=frontend, xmods=[tmp_path])

    expected_dependencies = {
        '#driver': ('kernel1_mod#kernel', 'kernel2_mod#kernel'),
        'kernel1_mod#kernel': ('kernel1_impl', 'kernel1_impl#kernel_impl'),
        'kernel1_impl': (),
        'kernel1_impl#kernel_impl': (),
        'kernel2_mod#kernel': ('kernel2_impl', 'kernel2_impl#kernel_impl'),
        'kernel2_impl': (),
        'kernel2_impl#kernel_impl': (),
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items()
        for b in deps
    }

    cg_path = tmp_path/'callgraph_scopes'
    scheduler.callgraph(cg_path)

    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {n.upper() for n in expected_dependencies}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in expected_dependencies.items()
        for b in deps
    }

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_typebound(tmp_path, testdir, config, frontend, proj_typebound_dependencies):
    """
    Test correct dependency chasing for typebound procedure calls.

    projTypeBound: driver -> some_type%other_routine -> other_routine -> some_type%routine1 -> routine1
                 | | | | | |                                          |                                |
                 | | | | | |       +- routine <- some_type%routine2 <-+                                +---------+
                 | | | | | |       |                                                                             |
                 | | | | | +--> some_type%some_routine -> some_routine -> some_type%routine -> module_routine  <-+
                 | | | +------> header_type%member_routine -> header_member_routine
                 | | +--------> header_type%routine -> header_type%routine_real -> header_routine_real
                 | |                           |
                 | |                           +---> header_type%routine_integer -> routine_integer
                 | +---------->other_type%member -> other_member -> header_member_routine   <--+
                 |                                                                             |
                 +------------>other_type%var%%member_routine -> header_type%member_routine  --+
    """
    proj = testdir/'sources/projTypeBound'

    scheduler = Scheduler(
        paths=proj, seed_routines=['driver'], config=config,
        full_parse=False, frontend=frontend, xmods=[tmp_path]
    )

    assert set(scheduler.items) == set(proj_typebound_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    cg_path = tmp_path/'callgraph_typebound'
    scheduler.callgraph(cg_path)

    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {n.upper() for n in proj_typebound_dependencies}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_typebound_ignore(tmp_path, testdir, config, frontend, proj_typebound_dependencies):
    """
    Test correct dependency chasing for typebound procedure calls with ignore working for
    typebound procedures correctly.

    projTypeBound: driver -> some_type%other_routine -> other_routine -> some_type%routine1 -> routine1
                   | | | | |                                          |                                |
                   | | | | |       +- routine <- some_type%routine2 <-+                                +---------+
                   | | | | |       |                                                                             |
                   | | | | +--> some_type%some_routine -> some_routine -> some_type%routine -> module_routine  <-+
                   | | +------> header_type%member_routine -> header_member_routine
                   | +--------> header_type%routine -> header_type%routine_real -> header_routine_real
                   |                           |
                   |                           +---> header_type%routine_integer -> routine_integer
                   +---------->other_type%member -> other_member -> header_member_routine
    """
    proj = testdir/'sources/projTypeBound'

    config['default']['disable'] += [
        'some_type%some_routine',
        'header_member_routine'
    ]
    config['routines'] = {
        'other_member': {
            'disable': config['default']['disable'] + ['member_routine']
        }
    }

    items_to_remove = (
        'typebound_item#some_type%some_routine',
        'typebound_item#some_routine',
        'typebound_item#some_type%routine',
        'typebound_header#header_member_routine',
    )

    proj_typebound_dependencies = {
        name: tuple(dep for dep in deps if dep not in items_to_remove)
        for name, deps in proj_typebound_dependencies.items()
        if name not in items_to_remove
    }

    scheduler = Scheduler(
        paths=proj, seed_routines=['driver'], config=config,
        full_parse=False, frontend=frontend, xmods=[tmp_path]
    )

    assert set(scheduler.items) == set(proj_typebound_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    cg_path = tmp_path/'callgraph_typebound'
    scheduler.callgraph(cg_path)

    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {n.upper() for n in proj_typebound_dependencies}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_nested_type_enrichment(tmp_path, frontend, config):
    """
    Make sure that enrichment works correctly for nested types across
    multiple source files
    """
    fcode1 = """
module typebound_procedure_calls_mod
    implicit none

    type my_type
        integer :: val
    contains
        procedure :: reset
        procedure :: add => add_my_type
    end type my_type

    type other_type
        type(my_type) :: arr(3)
    contains
        procedure :: add => add_other_type
        procedure :: total_sum
    end type other_type

contains

    subroutine reset(this)
        class(my_type), intent(inout) :: this
        this%val = 0
    end subroutine reset

    subroutine add_my_type(this, val)
        class(my_type), intent(inout) :: this
        integer, intent(in) :: val
        this%val = this%val + val
    end subroutine add_my_type

    subroutine add_other_type(this, other)
        class(other_type) :: this
        type(other_type) :: other
        integer :: i
        do i=1,3
            call this%arr(i)%add(other%arr(i)%val)
        end do
    end subroutine add_other_type

    function total_sum(this) result(result)
        class(other_type), intent(in) :: this
        integer :: result
        integer :: i
        result = 0
        do i=1,3
            result = result + this%arr(i)%val
        end do
    end function total_sum

end module typebound_procedure_calls_mod
    """.strip()

    fcode2 = """
module other_typebound_procedure_calls_mod
    use typebound_procedure_calls_mod, only: other_type
    implicit none

    type third_type
        type(other_type) :: stuff(2)
    contains
        procedure :: init
        procedure :: print => print_content
    end type third_type

contains

    subroutine init(this)
        class(third_type), intent(inout) :: this
        integer :: i, j
        do i=1,2
            do j=1,3
                call this%stuff(i)%arr(j)%reset()
                call this%stuff(i)%arr(j)%add(i+j)
            end do
        end do
    end subroutine init

    subroutine print_content(this)
        class(third_type), intent(inout) :: this
        call this%stuff(1)%add(this%stuff(2))
        print *, this%stuff(1)%total_sum()
    end subroutine print_content
end module other_typebound_procedure_calls_mod
    """.strip()

    fcode3 = """
subroutine driver
    use other_typebound_procedure_calls_mod, only: third_type
    implicit none
    type(third_type) :: data
    integer :: mysum

    call data%init()
    call data%stuff(1)%arr(1)%add(1)
    mysum = data%stuff(1)%total_sum() + data%stuff(2)%total_sum()
    call data%print
end subroutine driver
    """.strip()

    (tmp_path/'typebound_procedure_calls_mod.F90').write_text(fcode1)
    (tmp_path/'other_typebound_procedure_calls_mod.F90').write_text(fcode2)
    (tmp_path/'driver.F90').write_text(fcode3)

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['driver'],
        frontend=frontend, xmods=[tmp_path]
    )

    driver = scheduler['#driver'].source['driver']
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 3
    for call in calls:
        assert isinstance(call.name, ProcedureSymbol)
        assert isinstance(call.name.type.dtype, ProcedureType)
        assert call.name.parent
        assert isinstance(call.name.parent.type.dtype, DerivedType)
        assert isinstance(call.routine, Subroutine)
        assert isinstance(call.name.type.dtype.procedure, Subroutine)

    assert isinstance(calls[0].name.parent, Scalar)
    assert calls[0].name.parent.type.dtype.name == 'third_type'
    assert isinstance(calls[0].name.parent.type.dtype.typedef, ir.TypeDef)

    assert isinstance(calls[1].name.parent, Array)
    assert calls[1].name.parent.type.dtype.name == 'my_type'
    assert isinstance(calls[1].name.parent.type.dtype.typedef, ir.TypeDef)

    assert isinstance(calls[1].name.parent.parent, Array)
    assert isinstance(calls[1].name.parent.parent.type.dtype, DerivedType)
    assert calls[1].name.parent.parent.type.dtype.name == 'other_type'
    assert isinstance(calls[1].name.parent.parent.type.dtype.typedef, ir.TypeDef)

    assert isinstance(calls[1].name.parent.parent.parent, Scalar)
    assert isinstance(calls[1].name.parent.parent.parent.type.dtype, DerivedType)
    assert calls[1].name.parent.parent.parent.type.dtype.name == 'third_type'
    assert isinstance(calls[1].name.parent.parent.parent.type.dtype.typedef, ir.TypeDef)

    inline_calls = FindInlineCalls().visit(driver.body)
    assert len(inline_calls) == 2
    for call in inline_calls:
        assert isinstance(call.function, ProcedureSymbol)
        assert isinstance(call.function.type.dtype, ProcedureType)

        assert call.function.parent
        assert isinstance(call.function.parent, Array)
        assert isinstance(call.function.parent.type.dtype, DerivedType)
        assert call.function.parent.type.dtype.name == 'other_type'
        assert isinstance(call.function.parent.type.dtype.typedef, ir.TypeDef)

        assert call.function.parent.parent
        assert isinstance(call.function.parent.parent, Scalar)
        assert isinstance(call.function.parent.parent.type.dtype, DerivedType)
        assert call.function.parent.parent.type.dtype.name == 'third_type'
        assert isinstance(call.function.parent.parent.type.dtype.typedef, ir.TypeDef)


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_interface_inline_call(tmp_path, testdir, config, frontend):
    """
    Test that inline function calls declared via an explicit interface are added as dependencies.
    """

    my_config = config.copy()
    my_config['routines'] = {
        'driver': {
            'role': 'driver',
        }
    }

    scheduler = Scheduler(
        paths=testdir/'sources/projInlineCalls', config=my_config, frontend=frontend, xmods=[tmp_path]
    )

    expected_dependencies = {
        '#driver': (
            '#double_real', 'some_module', 'some_module#add_args', 'some_module#return_one',
            'some_module#some_type', 'some_module#some_type%do_something', 'vars_module',
        ),
        '#double_real': ('vars_module',),
        'some_module': (),
        'some_module#add_args': ('some_module#add_two_args', 'some_module#add_three_args'),
        'some_module#add_two_args': (),
        'some_module#add_three_args': (),
        'some_module#return_one': (),
        'some_module#some_type': (),
        'some_module#some_type%do_something': ('some_module#add_const',),
        'some_module#add_const': ('some_module#some_type',),
        'vars_module': (),
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    assert isinstance(scheduler['some_module#add_args'], InterfaceItem)
    assert isinstance(scheduler['#double_real'], ProcedureItem)
    assert isinstance(scheduler['some_module#some_type'], TypeDefItem)
    assert isinstance(scheduler['some_module#add_two_args'], ProcedureItem)
    assert isinstance(scheduler['some_module#add_three_args'], ProcedureItem)

    cg_path = tmp_path/'callgraph'
    scheduler.callgraph(cg_path)

    # Testing of callgraph visualisation with imports
    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {i.upper() for i in expected_dependencies}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in expected_dependencies.items() for b in deps
    }


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_interface_dependencies(tmp_path, frontend, config):
    """
    Ensure that interfaces are treated as intermediate nodes and incur
    dependencies on the actual procedures
    """
    fcode_module = """
module test_scheduler_interface_dependencies_mod
    implicit none
    interface my_intf
        procedure proc1
        procedure proc2
    end interface my_intf
contains
    subroutine proc1(arg)
        integer, intent(inout) :: arg
        arg = arg + 1
    end subroutine proc1
    subroutine proc2(arg)
        real, intent(inout) :: arg
        arg = arg + 1.0
    end subroutine proc2
end module test_scheduler_interface_dependencies_mod
    """
    fcode_driver = """
subroutine test_scheduler_interface_dependencies_driver
    use test_scheduler_interface_dependencies_mod, only: my_intf
    implicit none
    integer i
    real a
    i = 0
    a = 0.0
    call my_intf(i)
    call my_intf(a)
end subroutine test_scheduler_interface_dependencies_driver
    """

    config['routines']['test_scheduler_interface_dependencies_driver'] = {
        'role': 'driver'
    }

    (tmp_path/'test_scheduler_interface_dependencies_mod.F90').write_text(fcode_module)
    (tmp_path/'test_scheduler_interface_dependencies_driver.F90').write_text(fcode_driver)

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['test_scheduler_interface_dependencies_driver'],
        frontend=frontend, xmods=[tmp_path]
    )

    expected_dependencies = {
        '#test_scheduler_interface_dependencies_driver': {
            'test_scheduler_interface_dependencies_mod#my_intf'
        },
        'test_scheduler_interface_dependencies_mod#my_intf': {
            'test_scheduler_interface_dependencies_mod#proc1', 'test_scheduler_interface_dependencies_mod#proc2'
        },
        'test_scheduler_interface_dependencies_mod#proc1': set(),
        'test_scheduler_interface_dependencies_mod#proc2': set()
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    assert isinstance(scheduler['test_scheduler_interface_dependencies_mod#my_intf'], InterfaceItem)
    assert isinstance(scheduler['test_scheduler_interface_dependencies_mod#proc1'], ProcedureItem)
    assert isinstance(scheduler['test_scheduler_interface_dependencies_mod#proc2'], ProcedureItem)


@pytest.mark.parametrize('full_parse', [True, False])
def test_scheduler_typebound_inline_call(tmp_path, config, full_parse):
    fcode_mod = """
module some_mod
    implicit none
    type some_type
        integer :: a
    contains
        procedure :: some_routine
        procedure :: some_function
    end type some_type
contains
    subroutine some_routine(t)
        class(some_type), intent(inout) :: t
        t%a = 5
    end subroutine some_routine

    integer function some_function(t)
        class(some_type), intent(in) :: t
        some_function = t%a
    end function some_function
end module some_mod
    """.strip()

    fcode_caller = """
subroutine caller(b)
    use some_mod, only: some_type
    implicit none
    integer, intent(inout) :: b
    type(some_type) :: t
    t%a = b
    call t%some_routine()
    b = t%some_function()
end subroutine caller
    """.strip()

    (tmp_path/'some_mod.F90').write_text(fcode_mod)
    (tmp_path/'caller.F90').write_text(fcode_caller)

    def verify_graph(scheduler, expected_dependencies):
        assert set(scheduler.items) == set(expected_dependencies)
        assert set(scheduler.dependencies) == {
            (a, b) for a, deps in expected_dependencies.items() for b in deps
        }

        assert all(item.source._incomplete is not full_parse for item in scheduler.items)

        cg_path = tmp_path/'callgraph'
        scheduler.callgraph(cg_path)

        # Testing of callgraph visualisation
        vgraph = VisGraphWrapper(cg_path)
        assert set(vgraph.nodes) == {n.upper() for n in expected_dependencies}
        assert set(vgraph.edges) == {
            (a.upper(), b.upper()) for a, deps in expected_dependencies.items()
            for b in deps
        }

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['caller'], full_parse=full_parse, xmods=[tmp_path]
    )

    expected_dependencies = {
        '#caller': (
            'some_mod#some_type',
            'some_mod#some_type%some_routine',
        ),
        'some_mod#some_type': (),
        'some_mod#some_type%some_routine': ('some_mod#some_routine',),
        'some_mod#some_routine': ('some_mod#some_type',),
    }

    if scheduler.full_parse:
        # Inline Calls can only be fully resolved in a full parse
        expected_dependencies['#caller'] += ('some_mod#some_type%some_function',)
        expected_dependencies['some_mod#some_type%some_function'] = ('some_mod#some_function',)
        expected_dependencies['some_mod#some_function'] = ('some_mod#some_type',)

    verify_graph(scheduler, expected_dependencies)


@pytest.mark.parametrize('full_parse', [False, True])
def test_scheduler_cycle(tmp_path, config, full_parse):
    fcode_mod = """
module some_mod
    implicit none
    type some_type
        integer :: a
    contains
        procedure :: proc => some_proc
        procedure :: other => some_other
    end type some_type
contains
    recursive subroutine some_proc(this, val, recurse, fallback)
        class(some_type), intent(inout) :: this
        integer, intent(in) :: val
        logical, intent(in), optional :: recurse

        if (present(recurse)) then
            if (present(fallback)) then
                call this%other(val)
            else
                call some_proc(this, val, .true., .true.)
            end if
        else
            call this%proc(val, .true.)
        end if
    end subroutine some_proc

    subroutine some_other(this, val)
        class(some_type), intent(inout) :: this
        integer, intent(in) :: val
        this%a = val
    end subroutine some_other
end module some_mod
    """.strip()

    fcode_caller = """
subroutine caller
    use some_mod, only: some_type
    implicit none
    type(some_type) :: t

    call t%proc(1)
end subroutine caller
    """.strip()

    (tmp_path/'some_mod.F90').write_text(fcode_mod)
    (tmp_path/'caller.F90').write_text(fcode_caller)

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['caller'], full_parse=full_parse, xmods=[tmp_path]
    )

    # Make sure we the outgoing edges from the recursive routine to the procedure binding
    # and itself are removed but the other edge still exists
    assert (scheduler['#caller'], scheduler['some_mod#some_type%proc']) in scheduler.dependencies
    assert (scheduler['some_mod#some_type%proc'], scheduler['some_mod#some_proc']) in scheduler.dependencies
    assert (scheduler['some_mod#some_proc'], scheduler['some_mod#some_type%proc']) not in scheduler.dependencies
    assert (scheduler['some_mod#some_proc'], scheduler['some_mod#some_proc']) not in scheduler.dependencies
    assert (scheduler['some_mod#some_proc'], scheduler['some_mod#some_type%other']) in scheduler.dependencies
    assert (scheduler['some_mod#some_type%other'], scheduler['some_mod#some_other']) in scheduler.dependencies


def test_scheduler_unqualified_imports(config):
    """
    Test that only qualified imports are added as children.
    """

    kernel = """
    subroutine kernel()
       use some_mod
       use other_mod, only: other_routine

       call other_routine
    end subroutine kernel
    """

    source = Sourcefile.from_source(kernel, frontend=REGEX)
    item = ProcedureItem(name='#kernel', source=source, config=config['default'])

    assert len(item.dependencies) == 3
    children = set()
    for dep in item.dependencies:
        if isinstance(dep, ir.Import):
            if dep.symbols:
                children |= {f'{dep.module}#{str(s)}'.lower() for s in dep.symbols}
            else:
                children.add(dep.module.lower())
        elif isinstance(dep, ir.CallStatement):
            children.add(str(dep.name).lower())
        else:
            assert False, 'Unexpected dependency type'
    assert children == {'some_mod', 'other_mod#other_routine', 'other_routine'}


@pytest.mark.parametrize('frontend', available_frontends(skip={OMNI: "OMNI fails on missing module"}))
@pytest.mark.parametrize('enable_imports', [False, True])
@pytest.mark.parametrize('import_level', ['module', 'subroutine'])
def test_scheduler_indirect_import(frontend, tmp_path, enable_imports, import_level):
    fcode_mod_a = """
module a_mod
    implicit none
    public
    integer :: global_a = 1
end module a_mod
"""

    fcode_mod_b = """
module b_mod
    use a_mod
    implicit none
    public
    type type_b
        integer :: val
    end type type_b
end module b_mod
"""

    module_import_stmt = ""
    routine_import_stmt = ""
    if import_level == 'module':
        module_import_stmt = "use b_mod, only: type_b, global_a"
    elif import_level == 'subroutine':
        routine_import_stmt = "use b_mod, only: type_b, global_a"

    fcode_mod_c = f"""
module c_mod
    {module_import_stmt}
    implicit none
contains
    subroutine c(b)
        {routine_import_stmt}
        implicit none
        type(type_b), intent(inout) :: b
        b%val = global_a
    end subroutine c
end module c_mod
"""

    # Set-up paths and write sources
    src_path = tmp_path/'src'
    src_path.mkdir()
    out_path = tmp_path/'build'
    out_path.mkdir()

    (src_path/'a.F90').write_text(fcode_mod_a)
    (src_path/'b.F90').write_text(fcode_mod_b)
    (src_path/'c.F90').write_text(fcode_mod_c)

    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': enable_imports
        },
        'routines': {'c': {'role': 'driver'}}
    })
    # Create the Scheduler
    scheduler = Scheduler(
        paths=[src_path], config=config, frontend=frontend,
        output_dir=out_path, xmods=[out_path]
    )

    # Check for all items in the dependency graph
    expected_items = {'a_mod', 'b_mod', 'b_mod#type_b', 'c_mod#c'}
    assert expected_items == {item.name for item in scheduler.items}

    type_b = scheduler['b_mod#type_b'].ir
    c_mod_c = scheduler['c_mod#c'].ir
    var_map = CaseInsensitiveDict(
        (v.name, v) for v in FindVariables().visit(c_mod_c.body)
    )
    global_a = var_map['global_a']
    b_dtype = var_map['b'].type.dtype

    # Verify the type information for the imported symbols:
    # They will have enriched information if the imports are enabled
    # and deferred type otherwise
    if enable_imports:
        assert global_a.type.dtype is BasicType.INTEGER
        assert global_a.type.initial == '1'
        assert b_dtype.typedef is type_b
    else:
        assert global_a.type.dtype is BasicType.DEFERRED
        assert global_a.type.initial is None
        assert b_dtype.typedef is BasicType.DEFERRED


@pytest.mark.parametrize('frontend', available_frontends(skip={OMNI: "OMNI fails on missing module"}))
@pytest.mark.parametrize('external_kernel', [None, 'module', 'intfb'])
def test_scheduler_ignore_external_item(frontend, tmp_path, external_kernel):
    fcode_driver = f"""
module driver_mod
  contains
  subroutine driver(nlon, klev, nb, ydml_phy_mf)
    use parkind1, only: jpim, jprb
    use kernel1_mod, only: kernel1
    {'use kernel2_mod, only: kernel2' if external_kernel == 'module' else ''}
    implicit none
    type(model_physics_mf_type), intent(in) :: ydml_phy_mf
    integer(kind=jpim), intent(in) :: nlon
    integer(kind=jpim), intent(in) :: klev
    integer(kind=jpim), intent(in) :: nb
    integer(kind=jpim) :: jstart
    integer(kind=jpim) :: jend
    integer(kind=jpim) :: b
{'#include "kernel2.intfb.h"' if external_kernel == 'intfb' else ''}
    jstart = 1
    jend = nlon
    do b = 1, nb
        call kernel1()
        {'call kernel2()' if external_kernel else ''}
    enddo
  end subroutine driver
end module driver_mod
    """.strip()
    fcode_kernel1 = """
module kernel1_mod
  contains
  subroutine kernel1()
    use parkind1, only: jpim, jprb
  end subroutine kernel1
end module kernel1_mod
    """.strip()

    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel1_mod.F90').write_text(fcode_kernel1)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'ignore': ['parkind1'],
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }

    if external_kernel:
        config['default']['generated'] = ['kernel2*']

    class Trafo(Transformation):

        item_filter = (ProcedureItem, ModuleItem)

        def transform_module(self, module, **kwargs):
            pass

        def transform_subroutine(self, routine, **kwargs):
            pass

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        definitions=(), xmods=[tmp_path], frontend=frontend
    )

    expected_items = {'driver_mod#driver', 'parkind1', 'kernel1_mod#kernel1'}
    if external_kernel == 'module':
        expected_items.add('kernel2_mod#kernel2')
        expected_items.add('kernel2_mod')
    elif external_kernel == 'intfb':
        expected_items.add('#kernel2')

    assert expected_items == {item.name for item in scheduler.items}
    for item in scheduler.items:
        if item.name == 'parkind1':
            assert item.is_ignored
            assert isinstance(item, ExternalItem)
        if item.name.endswith('#kernel2'):
            assert not item.is_ignored
            assert isinstance(item, ExternalItem)
            assert item.is_generated

    if external_kernel:
        scheduler.process(transformation=Trafo(), proc_strategy=ProcessingStrategy.PLAN)
        # this shouldn't fail because we marked the item as build-time generated
        with pytest.raises(RuntimeError):
            scheduler.process(transformation=Trafo())
    else:
        # check whether this works without any error
        scheduler.process(transformation=Trafo())


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('qualified_import', ['driver', 'module', 'both', 'none'])
def test_scheduler_transient_typedef_imports(frontend, qualified_import, tmp_path):
    """ Test that use of transiently imported typedefs succeeds. """

    fcode_mod_a = f"""
module mod_a
    use mod_b{', only: my_type' if qualified_import in ('module', 'both') else ''}
    implicit none
end module mod_a
"""

    fcode_mod_b = """
module mod_b
    implicit none
    type my_type
        real(kind=4) :: a, b, x

        contains
        procedure :: add_a_b => my_type_add_a_b
    end type my_type

contains
    subroutine my_type_add_a_b(obj)
        type(my_type), intent(inout) :: obj

        obj%x = obj%a + obj%b
    end subroutine my_type_add_a_b
end module mod_b
"""

    fcode_driver = f"""
subroutine test_scheduler()
    use mod_a {', only: my_type' if qualified_import in ('driver', 'both') else ''}
    implicit none

    type(my_type) :: d

    d%a = 42.0
    d%a = 66.6
    call d%add_a_b()
end subroutine test_scheduler
"""
    src_path = tmp_path/'src'
    src_path.mkdir()
    (src_path/'mod_a.F90').write_text(fcode_mod_a)
    (src_path/'mod_b.F90').write_text(fcode_mod_b)
    (src_path/'driver.F90').write_text(fcode_driver)

    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel', 'expand': True, 'strict': True, 'enable_imports': True
        },
        'routines': {'test_scheduler': {'role': 'driver'}}
    })
    # Create the Scheduler
    if qualified_import not in ('driver', 'both'):
        with pytest.raises(RuntimeError):
            _ = Scheduler(
                paths=[src_path], config=config, seed_routines='test_scheduler',
                frontend=frontend, xmods=[tmp_path], full_parse=False
            )
    else:
        scheduler = Scheduler(
            paths=[src_path], config=config, seed_routines='test_scheduler',
            frontend=frontend, xmods=[tmp_path], full_parse=False
        )

        if qualified_import == 'both':
            assert scheduler.items == (
                '#test_scheduler', 'mod_a', 'mod_a#my_type%add_a_b', 'mod_b#my_type'
            )
            assert scheduler.dependencies == (
                ('#test_scheduler', 'mod_a'),
                ('#test_scheduler', 'mod_a#my_type%add_a_b'),
                ('mod_a', 'mod_b#my_type')
            )
        else:
            assert scheduler.items == (
                '#test_scheduler', 'mod_a', 'mod_a#my_type%add_a_b', 'mod_b'
            )
            assert scheduler.dependencies == (
                ('#test_scheduler', 'mod_a'),
                ('#test_scheduler', 'mod_a#my_type%add_a_b'),
                ('mod_a', 'mod_b')
            )

        if frontend == OMNI:
            # OMNI fails to read due to missing mod_b xmods
            with pytest.raises(ParseError):
                scheduler._parse_items()
        else:
            scheduler._parse_items()

        call = FindNodes(ir.CallStatement).visit(scheduler['#test_scheduler'].ir.ir)[0]
        assert call.name == 'd%add_a_b'
        assert isinstance(call.name.parent.type.dtype, DerivedType)
        assert call.name.parent.type.dtype.name == 'my_type'

        # Enrichment does work correctly in this situation, providing the link to the typedef
        if qualified_import in ('driver', 'both') and frontend == FP:
            assert call.name.parent.type.dtype.typedef is (
                getattr(scheduler['mod_b#my_type'], 'ir', None) or scheduler['mod_b'].ir['my_type']
            )
        else:
            # The interprocedural annotations do _not_ currently enrich the type in this situation
            assert call.name.parent.type.dtype.typedef == BasicType.DEFERRED


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('qualified_import', ['driver', 'module', 'both', 'none'])
def test_scheduler_transient_procedure_imports(frontend, qualified_import, tmp_path):
    """ Test that use of transiently imported procedures succeeds. """

    fcode_mod_a = f"""
module mod_a
    use mod_b{', only: my_proc' if qualified_import in ('module', 'both') else ''}
    implicit none
end module mod_a
"""

    fcode_mod_b = """
module mod_b
    implicit none
contains
    subroutine my_proc
        print *,'hello world'
    end subroutine
end module mod_b
"""

    fcode_driver = f"""
subroutine test_scheduler()
    use mod_a {', only: my_proc' if qualified_import in ('driver', 'both') else ''}
    implicit none

    call my_proc
end subroutine test_scheduler
"""
    src_path = tmp_path/'src'
    src_path.mkdir()
    (src_path/'mod_a.F90').write_text(fcode_mod_a)
    (src_path/'mod_b.F90').write_text(fcode_mod_b)
    (src_path/'driver.F90').write_text(fcode_driver)

    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel', 'expand': True, 'strict': True, 'enable_imports': True
        },
        'routines': {'test_scheduler': {'role': 'driver'}}
    })

    # Create the Scheduler
    if qualified_import in ('module', 'none'):
        with pytest.raises(RuntimeError):
            _ = Scheduler(
                paths=[src_path], config=config, seed_routines='test_scheduler',
                frontend=frontend, xmods=[tmp_path], full_parse=False
            )
    else:
        scheduler = Scheduler(
            paths=[src_path], config=config, seed_routines='test_scheduler',
            frontend=frontend, xmods=[tmp_path], full_parse=False
        )

        if qualified_import == 'both':
            assert scheduler.items == ('#test_scheduler', 'mod_a', 'mod_a#my_proc')
            assert scheduler.dependencies == (
                ('#test_scheduler', 'mod_a'), ('#test_scheduler', 'mod_a#my_proc')
            )
        else:
            assert scheduler.items == ('#test_scheduler', 'mod_a', 'mod_a#my_proc', 'mod_b')
            assert scheduler.dependencies == (
                ('#test_scheduler', 'mod_a'), ('#test_scheduler', 'mod_a#my_proc'),
                ('mod_a', 'mod_b')
            )

        if frontend == OMNI and qualified_import == 'both':
            # OMNI fails to read due to missing mod_b xmods
            with pytest.raises(Exception):
                scheduler._parse_items()
        else:
            scheduler._parse_items()

        call = FindNodes(ir.CallStatement).visit(scheduler['#test_scheduler'].ir.ir)[0]
        # The interprocedural annotations do _not_ currently enrich the call in this situation
        assert call.name == 'my_proc'
        assert call.name.type.imported
        assert not call.name.type.procedure


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('qualified_import', ['driver', 'module', 'both', 'none'])
def test_scheduler_transient_variable_imports(frontend, qualified_import, tmp_path):
    """ Test that use of transiently imported variables succeeds. """

    fcode_mod_a = f"""
module mod_a
    use mod_b{', only: my_var' if qualified_import in ('module', 'both') else ''}
    implicit none
end module mod_a
"""

    fcode_mod_b = """
module mod_b
    implicit none
    integer :: my_var
end module mod_b
"""

    fcode_driver = f"""
subroutine test_scheduler()
    use mod_a {', only: my_var' if qualified_import in ('driver', 'both') else ''}
    implicit none
    my_var = 1
end subroutine test_scheduler
"""
    src_path = tmp_path/'src'
    src_path.mkdir()
    (src_path/'mod_a.F90').write_text(fcode_mod_a)
    (src_path/'mod_b.F90').write_text(fcode_mod_b)
    (src_path/'driver.F90').write_text(fcode_driver)

    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel', 'expand': True, 'strict': True, 'enable_imports': True
        },
        'routines': {'test_scheduler': {'role': 'driver'}}
    })

    scheduler = Scheduler(
        paths=[src_path], config=config, seed_routines='test_scheduler',
        frontend=frontend, xmods=[tmp_path], full_parse=False
    )

    assert scheduler.items == ('#test_scheduler', 'mod_a', 'mod_b')
    assert scheduler.dependencies == (
        ('#test_scheduler', 'mod_a'), ('mod_a', 'mod_b')
    )

    scheduler._parse_items()

    my_var = scheduler['#test_scheduler'].ir.body.body[0].lhs
    # NB: Global variable imports as a dependency are established directly on the import
    #     statements, thus the transient dependency is captured
    assert isinstance(my_var, Scalar)
    assert my_var.type.dtype == BasicType.INTEGER
    assert my_var.type.imported
    assert my_var.type.module is scheduler['mod_a'].ir

    if qualified_import in ('module', 'both'):
        mod_a_var = scheduler['mod_a'].ir.imported_symbol_map['my_var']
        # NB: We are able to correctly propagate the type information through
        #     multiple import layers
        assert isinstance(mod_a_var, Scalar)
        assert mod_a_var.type.dtype == BasicType.INTEGER
        assert mod_a_var.type.imported
        # NB: This links to the module it imports from, not where it is defined
        assert mod_a_var.type.module is scheduler['mod_b'].ir


def test_scheduler_module_interface_import(frontend, tmp_path):
    """ Test module-level imports of interface routines. """

    fcode_mod_a = """
module mod_a
    use mod_b, only: inner_type, intfb_routine
    implicit none
    type outer_type
        type(inner_type) :: da
    end type outer_type
end module mod_a
"""

    fcode_mod_b = """
module mod_b
    implicit none
    type inner_type
        integer :: da, db
    end type inner_type

    interface intfb_routine
    module procedure routine_a, routine_b
    end interface intfb_routine

    contains
    subroutine routine_a
    end subroutine routine_a

    subroutine routine_b
    end subroutine routine_b
end module mod_b
"""

    fcode_driver = """
subroutine test_scheduler()
    use mod_a, only: outer_type
    implicit none
    type(outer_type) :: da

    da%da%da = 42.0
end subroutine test_scheduler
"""
    src_path = tmp_path/'src'
    src_path.mkdir()
    (src_path/'mod_a.F90').write_text(fcode_mod_a)
    (src_path/'mod_b.F90').write_text(fcode_mod_b)
    (src_path/'driver.F90').write_text(fcode_driver)

    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel', 'expand': True, 'strict': True, 'enable_imports': True
        },
        'routines': {'test_scheduler': {'role': 'driver'}}
    })

    scheduler = Scheduler(
        paths=[src_path], config=config, seed_routines='test_scheduler',
        frontend=frontend, xmods=[tmp_path], full_parse=False
    )

    assert scheduler.items == (
        '#test_scheduler', 'mod_a#outer_type', 'mod_b#inner_type'
    )
    assert scheduler.dependencies == (
        ('#test_scheduler', 'mod_a#outer_type'),
        ('mod_a#outer_type', 'mod_b#inner_type')
    )
