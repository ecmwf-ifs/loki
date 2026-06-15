# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from functools import partial
import re
from pathlib import Path
from subprocess import CalledProcessError

import pytest

from loki.logging import WARNING
from loki import Array, Dimension, FindNodes, Literal, Sourcefile
from loki.batch import Pipeline, ProcessingStrategy, Scheduler, SchedulerConfig, Transformation
from loki.frontend import HAVE_FP, HAVE_OMNI, OMNI, available_frontends
from loki.ir import nodes as ir
from loki.transformations import (
    CMakePlanTransformation, DependencyTransformation, FileWriteTransformation,
    ModuleWrapTransformation
)

pytestmark = pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')


def test_scheduler_empty_config(testdir, frontend, tmp_path):
    """
    Test that instantiating the Scheduler without config works (albeit it's not very useful)
    This fixes #373
    """
    projA = testdir/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include',
        seed_routines=['driverA'], frontend=frontend, xmods=[tmp_path]
    )
    assert scheduler.items == ('drivera_mod#drivera',)


def test_scheduler_cmake_planner(tmp_path, testdir, frontend):
    """
    Test the plan generation feature over a call hierarchy spanning two
    distinctive projects.

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """

    sourcedir = testdir/'sources'
    proj_a = sourcedir/'projA'
    proj_b = sourcedir/'projB'

    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'ignore': ('header_mod',),
            'mode': 'foobar'
        },
        'routines': {
            'driverB': {'role': 'driver'},
            'kernelB': {'ignore': ['ext_driver']},
        }
    })
    builddir = tmp_path/'scheduler_cmake_planner_dummy_dir'
    builddir.mkdir(exist_ok=True)

    scheduler = Scheduler(
        paths=[proj_a, proj_b], includes=proj_a/'include',
        config=config, frontend=frontend, xmods=[tmp_path],
        output_dir=builddir
    )

    planfile = builddir/'loki_plan.cmake'

    # Populate the scheduler
    # (this is the same as SchedulerA in test_scheduler_dependencies_ignore, so no need to
    # check scheduler set-up itself)
    # Apply the transformation
    scheduler.process(FileWriteTransformation(), proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=planfile, rootpath=sourcedir)

    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)

    loki_plan = planfile.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    # Validate the plan file content
    expected_files = {
        'driverB_mod', 'kernelB_mod',
        'compute_l1_mod', 'compute_l2_mod'
    }

    assert 'LOKI_SOURCES_TO_TRANSFORM' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == expected_files

    assert 'LOKI_SOURCES_TO_REMOVE' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == expected_files

    assert 'LOKI_SOURCES_TO_APPEND' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {
        f'{name}.foobar' for name in expected_files
    }

    planfile.unlink()
    builddir.rmdir()


@pytest.mark.parametrize('prec', ['DP', 'SP'])
def test_scheduler_cmake_planner_libs(tmp_path, testdir, frontend, prec):
    """
    Test the plan generation feature over a call hierarchy spanning two
    distinctive projects. However, this time using the 'lib' attribute.

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """

    sourcedir = testdir/'sources'
    proj_a = sourcedir/'projA'
    proj_b = sourcedir/'projB'

    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'ignore': ('header_mod',),
            'mode': 'foobar',
            'lib': f'projAlib.{prec}'
        },
        'routines': {
            'driverB': {'role': 'driver'},
            'ext_driver': {'lib': f'projBlib.{prec}'}
        }
    })
    builddir = tmp_path/'scheduler_cmake_planner_libs_dummy_dir'
    builddir.mkdir(exist_ok=True)

    scheduler = Scheduler(
        paths=[proj_a, proj_b], includes=proj_a/'include',
        config=config, frontend=frontend, xmods=[tmp_path],
        output_dir=builddir
    )

    planfile = builddir/'loki_plan_libs.cmake'
    # Populate the scheduler
    # Apply the transformation
    scheduler.process(FileWriteTransformation(), proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=planfile, rootpath=sourcedir)

    loki_plan = planfile.read_text()
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    # Validate the plan file content
    expected_keys = {
        'LOKI_SOURCES_TO_TRANSFORM', 'LOKI_SOURCES_TO_APPEND', 'LOKI_SOURCES_TO_REMOVE',
        f'LOKI_SOURCES_TO_TRANSFORM_projBlib_{prec}', f'LOKI_SOURCES_TO_APPEND_projBlib_{prec}',
        f'LOKI_SOURCES_TO_REMOVE_projBlib_{prec}', f'LOKI_SOURCES_TO_TRANSFORM_projAlib_{prec}',
        f'LOKI_SOURCES_TO_APPEND_projAlib_{prec}', f'LOKI_SOURCES_TO_REMOVE_projAlib_{prec}'
    }

    assert set(plan_dict.keys()) == expected_keys

    expected_files_a = {
        'driverB_mod', 'kernelB_mod',
        'compute_l1_mod', 'compute_l2_mod',
    }
    expected_files_b = {'ext_driver_mod', 'ext_kernel'}
    expected_files = expected_files_a | expected_files_b

    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == expected_files
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == expected_files
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {
        f'{name}.foobar' for name in expected_files
    }

    assert plan_dict[f'LOKI_SOURCES_TO_TRANSFORM_projAlib_{prec}'] == expected_files_a
    assert plan_dict[f'LOKI_SOURCES_TO_REMOVE_projAlib_{prec}'] == expected_files_a
    assert plan_dict[f'LOKI_SOURCES_TO_APPEND_projAlib_{prec}'] == {
        f'{name}.foobar' for name in expected_files_a
    }

    assert plan_dict[f'LOKI_SOURCES_TO_TRANSFORM_projBlib_{prec}'] == expected_files_b
    assert plan_dict[f'LOKI_SOURCES_TO_REMOVE_projBlib_{prec}'] == expected_files_b
    assert plan_dict[f'LOKI_SOURCES_TO_APPEND_projBlib_{prec}'] == {
        f'{name}.foobar' for name in expected_files_b
    }

    planfile.unlink()
    builddir.rmdir()


def test_scheduler_disable_wildcard(testdir, config, tmp_path):
    fcode_mod = """
module field_mod
  type field2d
    contains
    procedure :: init => field_init
  end type

  type field3d
    contains
    procedure :: init => field_init
  end type

  contains
    subroutine field_init()

    end subroutine
end module
"""

    fcode_driver = """
subroutine my_driver
  use field_mod, only: field2d, field3d, field_init
implicit none

  type(field2d) :: a, b
  type(field3d) :: c, d

  call a%init()
  call b%init()
  call c%init()
  call field_init(d)
end subroutine my_driver
"""

    # Set up the test files
    dirname = testdir/'test_scheduler_disable_wildcard'
    dirname.mkdir(exist_ok=True)
    modfile = dirname/'field_mod.F90'
    modfile.write_text(fcode_mod)
    testfile = dirname/'test.F90'
    testfile.write_text(fcode_driver)

    config['default']['disable'] = ['*%init']

    scheduler = Scheduler(paths=dirname, seed_routines=['my_driver'], config=config, xmods=[tmp_path])

    expected_items = ['#my_driver', 'field_mod#field_init']
    expected_dependencies = [('#my_driver', 'field_mod#field_init')]

    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    assert 'field_mod#field2d%init' not in scheduler.items
    assert 'field_mod#field3d%init' not in scheduler.items

    try:
        # Clean up
        modfile.unlink()
        testfile.unlink()
        dirname.rmdir()
    except FileNotFoundError:
        pass


def test_transformation_config(config):
    """
    Test the correct instantiation of :any:`Transformation` objecst from config
    """
    my_config = config.copy()
    my_config['transformations'] = {
        'DependencyTransformation': {
            'module': 'loki.transformations.build_system',
            'options': {
                'suffix': '_rick',
                'module_suffix': '_roll',
                'replace_ignore_items': False,
            }
        },
        'IdemTransformation': {
            'module': 'loki.transformations',
        }
    }
    cfg = SchedulerConfig.from_dict(my_config)
    assert cfg.transformations['DependencyTransformation']

    # Instantiate IdemTransformation entry without options
    transformation = cfg.transformations['DependencyTransformation']
    assert isinstance(transformation, DependencyTransformation)
    assert transformation.suffix == '_rick'
    assert transformation.module_suffix == '_roll'
    assert not transformation.replace_ignore_items

    bad_config = config.copy()
    bad_config['transformations'] = {
        'DependencyTrafo': {
            'module': 'loki.transformations.build_system',
            'options': {}
        }
    }
    # Test for errors when failing to instantiate a transformation
    with pytest.raises(RuntimeError):
        SchedulerConfig.from_dict(bad_config)

    worse_config = config.copy()
    worse_config['transformations'] = {
        'DependencyTransform': {
            # <= typo
            'module': 'loki.transformats.build_system',
            'options': {}
        }
    }
    with pytest.raises(ModuleNotFoundError):
        SchedulerConfig.from_dict(worse_config)

    worst_config = config.copy()
    worst_config['transformations'] = {
        'DependencyTransform': {
            'module': 'loki.transformations.build_system',
            # <= typo
            'options': {'hello': 'Dave'}
        }
    }
    with pytest.raises(RuntimeError):
        SchedulerConfig.from_dict(worst_config)


def test_transformation_config_external_with_dimension(testdir, config):
    """
    Test instantiation of :any:`Transformation` from config with
    :any:`Dimension` argument.
    """
    my_config = config.copy()
    my_config['dimensions'] = {
        'ij': {'size': 'n', 'index': 'i'}
    }
    my_config['transformations'] = {
        'CallMeRick': {
            'classname': 'CallMeMaybeTrafo',
            'module': 'call_me_trafo',
            'path': str(testdir/'sources'),
            'options': {'name': 'Rick', 'horizontal': '%dimensions.ij%'}
        }
    }
    cfg = SchedulerConfig.from_dict(my_config)
    assert cfg.transformations['CallMeRick']

    transformation = cfg.transformations['CallMeRick']
    # We don't have the type, so simply check the class name
    assert type(transformation).__name__ == 'CallMeMaybeTrafo'
    assert transformation.name == 'Rick'
    assert isinstance(transformation.horizontal, Dimension)
    assert transformation.horizontal.size == 'n'
    assert transformation.horizontal.index == 'i'


@pytest.mark.parametrize('item_name,keys,use_pattern_matching,match_item_parents,expected', [
    ('comp2', 'comp2', True, True, ('comp2',)),
    ('#comp2', 'comp2', True, True, ('comp2',)),
    ('comp2', '#comp2', True, True, ()),
    ('#comp2', '#comp2', True, True, ('#comp2',))
])
def test_scheduler_config_match_item_keys(item_name, keys, use_pattern_matching, match_item_parents, expected):
    # This is key: If the config key is provided with explicit scope,
    # we don't match unscoped names
    value = SchedulerConfig.match_item_keys(item_name, keys, use_pattern_matching, match_item_parents)
    assert value == expected


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('frontend_args,defines,preprocess,has_cpp_directives,additional_dependencies', [
    (
        None, None, False,
        ['#test_scheduler_frontend_args1', '#test_scheduler_frontend_args2', '#test_scheduler_frontend_args4'], {
        '#test_scheduler_frontend_args2': ('#test_scheduler_frontend_args3',),
        '#test_scheduler_frontend_args3': (),
        '#test_scheduler_frontend_args4': ('#test_scheduler_frontend_args3',),
        }
    ),
    (None, ['SOME_DEFINITION'], True, [], {}),
    ({'file3_4.F90': {'defines': ['SOME_DEFINITION', 'LOCAL_DEFINITION']}}, ['SOME_DEFINITION'], True, [], {
        '#test_scheduler_frontend_args3': (),
        '#test_scheduler_frontend_args4': ('#test_scheduler_frontend_args3',),
    }),
    ({'file2.F90': {'preprocess': False}}, ['SOME_DEFINITION'], True, ['#test_scheduler_frontend_args2'], {
        '#test_scheduler_frontend_args2': ('#test_scheduler_frontend_args3',),
        '#test_scheduler_frontend_args3': (),
    }),
    ({'file2.F90': {'preprocess': True, 'defines': ['SOME_DEFINITION']}}, None, False,
     ['#test_scheduler_frontend_args1', '#test_scheduler_frontend_args4'], {
         '#test_scheduler_frontend_args3': (),
         '#test_scheduler_frontend_args4': ('#test_scheduler_frontend_args3',),
     }),
])
def test_scheduler_frontend_args(tmp_path, frontend, frontend_args, defines, preprocess,
                                 has_cpp_directives, additional_dependencies, config):
    """
    Test overwriting frontend options via Scheduler config
    """

    fcode1 = """
subroutine test_scheduler_frontend_args1
    implicit none
#ifdef SOME_DEFINITION
    call test_scheduler_frontend_args2
#endif
end subroutine test_scheduler_frontend_args1
    """.strip()

    fcode2 = """
subroutine test_scheduler_frontend_args2
    implicit none
#ifndef SOME_DEFINITION
    call test_scheduler_frontend_args3
#endif
    call test_scheduler_frontend_args4
end subroutine test_scheduler_frontend_args2
    """.strip()

    fcode3_4 = """
subroutine test_scheduler_frontend_args3
implicit none
end subroutine test_scheduler_frontend_args3

subroutine test_scheduler_frontend_args4
implicit none
#ifdef LOCAL_DEFINITION
    call test_scheduler_frontend_args3
#endif
end subroutine test_scheduler_frontend_args4
    """.strip()

    (tmp_path/'file1.F90').write_text(fcode1)
    (tmp_path/'file2.F90').write_text(fcode2)
    (tmp_path/'file3_4.F90').write_text(fcode3_4)

    expected_dependencies = {
        '#test_scheduler_frontend_args1': ('#test_scheduler_frontend_args2',),
        '#test_scheduler_frontend_args2': ('#test_scheduler_frontend_args4',),
        '#test_scheduler_frontend_args4': (),
    }

    # No preprocessing, thus all call dependencies are included
    # Global preprocessing setting SOME_DEFINITION, removing dependency on 3
    # Global preprocessing with local definition for one file, re-adding a dependency on 3
    # Global preprocessing with preprocessing switched off for 2
    # No preprocessing except for 2
    for key, value in additional_dependencies.items():
        expected_dependencies[key] = expected_dependencies.get(key, ()) + value

    config['frontend_args'] = frontend_args

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['test_scheduler_frontend_args1'],
        frontend=frontend, defines=defines, preprocess=preprocess, xmods=[tmp_path]
    )

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    for item in scheduler.items:
        cpp_directives = FindNodes(ir.PreprocessorDirective).visit(item.ir.ir)
        # NB: OMNI always does preprocessing, therefore we won't find the CPP directives
        #     after the full parse
        assert bool(cpp_directives) == (item in has_cpp_directives and frontend != OMNI)


@pytest.mark.skipif(not (HAVE_OMNI and HAVE_FP), reason='OMNI or FP not available')
def test_scheduler_frontend_overwrite(tmp_path, config):
    """
    Test the use of a different frontend via Scheduler config
    """
    fcode_header = """
module test_scheduler_frontend_overwrite_header
    implicit none
    type some_type
        ! We have a comment
        real, dimension(:,:), pointer :: arr
    end type some_type
end module test_scheduler_frontend_overwrite_header
    """.strip()
    fcode_kernel = """
subroutine test_scheduler_frontend_overwrite_kernel
    use test_scheduler_frontend_overwrite_header, only: some_type
    implicit none
    type(some_type) :: var
end subroutine test_scheduler_frontend_overwrite_kernel
    """.strip()

    (tmp_path/'test_scheduler_frontend_overwrite_header.F90').write_text(fcode_header)
    (tmp_path/'test_scheduler_frontend_overwrite_kernel.F90').write_text(fcode_kernel)

    # Make sure that OMNI cannot parse the header file
    with pytest.raises(CalledProcessError):
        Sourcefile.from_source(fcode_header, frontend=OMNI, xmods=[tmp_path])

    # ...and that the problem exists also during Scheduler traversal
    with pytest.raises(CalledProcessError):
        Scheduler(
            paths=[tmp_path], config=config, seed_routines=['test_scheduler_frontend_overwrite_kernel'],
            frontend=OMNI, xmods=[tmp_path]
        )

    fcode_header_lines = fcode_header.split('\n')
    # Strip the comment from the header file and parse again to generate an xmod
    Sourcefile.from_source('\n'.join(fcode_header_lines[:3] + fcode_header_lines[4:]), frontend=OMNI, xmods=[tmp_path])

    # Setup the config with the frontend overwrite
    config['frontend_args'] = {
        'test_scheduler_frontend_overwrite_header.F90': {'frontend': 'FP'}
    }

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['test_scheduler_frontend_overwrite_kernel'],
        frontend=OMNI, xmods=[tmp_path]
    )

    # ...and now it works fine
    assert set(scheduler.items) == {
        '#test_scheduler_frontend_overwrite_kernel', 'test_scheduler_frontend_overwrite_header#some_type'
    }

    assert set(scheduler.dependencies) == {
        ('#test_scheduler_frontend_overwrite_kernel', 'test_scheduler_frontend_overwrite_header#some_type')
    }

    comments = FindNodes(ir.Comment).visit(scheduler['test_scheduler_frontend_overwrite_header#some_type'].ir.body)
    assert len(comments) == 1
    # ...and the derived type has it's comment
    assert comments[0].text == '! We have a comment'


def test_scheduler_pipeline_simple(testdir, config, frontend, tmp_path):
    """
    Test processing a :any:`Pipeline` over a simple call-tree.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = testdir/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines='driverA', frontend=frontend, xmods=[tmp_path]
    )

    class ZeroMyStuffTrafo(Transformation):
        """ Fill each argument array with 0.0 """

        def transform_subroutine(self, routine, **kwargs):
            for v in routine.variables:
                if isinstance(v, Array):
                    routine.body.append(ir.Assignment(lhs=v, rhs=Literal(0.0)))

    class AddSnarkTrafo(Transformation):
        """ Add a snarky comment to the zeroing """

        def __init__(self, name='Rick'):
            self.name = name

        def transform_subroutine(self, routine, **kwargs):
            # Add a newline
            routine.body.append(ir.Comment(text=''))
            routine.body.append(ir.Comment(text=f'! Sorry {self.name}, no values for you!'))

    def has_correct_assigns(routine, num_assign, values=None):
        assigns = FindNodes(ir.Assignment).visit(routine.body)
        values = values or [0.0]
        return len(assigns) == num_assign and all(a.rhs in values for a in assigns)

    def has_correct_comments(routine, name='Dave'):
        text = f'! Sorry {name}, no values for you!'
        comments = FindNodes(ir.Comment).visit(routine.body)
        return len(comments) > 2 and comments[-1].text == text

    # First apply in sequence and check effect
    scheduler.process(transformation=ZeroMyStuffTrafo())
    assert has_correct_assigns(scheduler['drivera_mod#drivera'].ir, 0)
    assert has_correct_assigns(scheduler['kernela_mod#kernela'].ir, 2)
    assert has_correct_assigns(scheduler['compute_l1_mod#compute_l1'].ir, 1)
    assert has_correct_assigns(scheduler['compute_l2_mod#compute_l2'].ir, 2, values=[66.0, 00])
    assert has_correct_assigns(scheduler['#another_l1'].ir, 1)
    assert has_correct_assigns(scheduler['#another_l2'].ir, 2, values=[77.0, 00])

    scheduler.process(transformation=AddSnarkTrafo(name='Dave'))
    assert has_correct_comments(scheduler['drivera_mod#drivera'].ir)
    assert has_correct_comments(scheduler['kernela_mod#kernela'].ir)
    assert has_correct_comments(scheduler['compute_l1_mod#compute_l1'].ir)
    assert has_correct_comments(scheduler['compute_l2_mod#compute_l2'].ir)
    assert has_correct_comments(scheduler['#another_l1'].ir)
    assert has_correct_comments(scheduler['#another_l2'].ir)

    # Rebuild the scheduler to wipe the previous result
    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines='driverA', frontend=frontend, xmods=[tmp_path]
    )

    my_pipeline = partial(Pipeline, classes=(ZeroMyStuffTrafo, AddSnarkTrafo))
    # Then apply as a simple pipeline and check again
    scheduler.process(transformation=my_pipeline(name='Chad'))
    assert has_correct_assigns(scheduler['drivera_mod#drivera'].ir, 0)
    assert has_correct_assigns(scheduler['kernela_mod#kernela'].ir, 2)
    assert has_correct_assigns(scheduler['compute_l1_mod#compute_l1'].ir, 1)
    assert has_correct_assigns(scheduler['compute_l2_mod#compute_l2'].ir, 2, values=[66.0, 00])
    assert has_correct_assigns(scheduler['#another_l1'].ir, 1)
    assert has_correct_assigns(scheduler['#another_l2'].ir, 2, values=[77.0, 00])

    assert has_correct_comments(scheduler['drivera_mod#drivera'].ir, name='Chad')
    assert has_correct_comments(scheduler['kernela_mod#kernela'].ir, name='Chad')
    assert has_correct_comments(scheduler['compute_l1_mod#compute_l1'].ir, name='Chad')
    assert has_correct_comments(scheduler['compute_l2_mod#compute_l2'].ir, name='Chad')
    assert has_correct_comments(scheduler['#another_l1'].ir, name='Chad')
    assert has_correct_comments(scheduler['#another_l2'].ir, name='Chad')


def test_pipeline_config_compose(config):
    """
    Test the correct instantiation of a custom :any:`Pipeline`
    object from config.
    """
    my_config = config.copy()
    my_config['dimensions'] = {
        'horizontal': {'size': 'KLON', 'index': 'JL', 'bounds': ['KIDIA', 'KFDIA']},
        'block_dim': {'size': 'NGPBLKS', 'index': 'IBL'},
    }
    my_config['transformations'] = {
        'VectorWithTrim': {
            'classname': 'SCCVectorPipeline',
            'module': 'loki.transformations.single_column',
            'options': {
                'horizontal': '%dimensions.horizontal%',
                'block_dim': '%dimensions.block_dim%',
                'directive': 'openacc',
                'trim_vector_sections': True,
            },
        },
        'preprocess': {
            'classname': 'RemoveCodeTransformation',
            'module': 'loki.transformations',
            'options': {
                'call_names': 'dr_hook',
                'remove_imports': False
            }
        },
        'postprocess': {
            'classname': 'ModuleWrapTransformation',
            'module': 'loki.transformations.build_system',
            'options': {'module_suffix': '_module'}
        }
    }
    my_config['pipelines'] = {
        'MyVectorPipeline': {
            'transformations': [
                'preprocess',
                'VectorWithTrim',
                'postprocess',
            ],
        }
    }
    cfg = SchedulerConfig.from_dict(my_config)

    # Check that transformations and pipelines were created correctly
    assert cfg.transformations['VectorWithTrim']
    assert cfg.transformations['preprocess']
    assert cfg.transformations['postprocess']

    assert cfg.pipelines['MyVectorPipeline']
    pipeline = cfg.pipelines['MyVectorPipeline']
    assert isinstance(pipeline, Pipeline)

    # Check that the pipeline is correctly composed
    assert len(pipeline.transformations) == 9
    assert type(pipeline.transformations[0]).__name__ == 'RemoveCodeTransformation'
    assert type(pipeline.transformations[1]).__name__ == 'SCCFuseVerticalLoops'
    assert type(pipeline.transformations[2]).__name__ == 'SCCBaseTransformation'
    assert type(pipeline.transformations[3]).__name__ == 'SCCDevectorTransformation'
    assert type(pipeline.transformations[4]).__name__ == 'SCCDemoteTransformation'
    assert type(pipeline.transformations[5]).__name__ == 'SCCVecRevectorTransformation'
    assert type(pipeline.transformations[6]).__name__ == 'SCCAnnotateTransformation'
    assert type(pipeline.transformations[7]).__name__ == 'PragmaModelTransformation'
    assert type(pipeline.transformations[8]).__name__ == 'ModuleWrapTransformation'

    # Check for some specified and default constructor flags
    assert pipeline.transformations[0].call_names == ('dr_hook',)
    assert pipeline.transformations[0].remove_imports is False
    assert isinstance(pipeline.transformations[2].horizontal, Dimension)
    assert pipeline.transformations[2].horizontal.size == 'KLON'
    assert pipeline.transformations[2].horizontal.index == 'JL'
    assert pipeline.transformations[3].trim_vector_sections is True
    assert pipeline.transformations[8].replace_ignore_items is True

@pytest.fixture(name='multi_modes_config')
def fixture_multi_modes_config():
    return {
        'default': {'role': 'kernel', 'expand': True, 'strict': False, 'mode': 'm1', 'replicate': True},
        'routines': {
            'driver_0': {'role': 'driver', 'mode': 'm1', 'replicate': False},
            'driver_1': {'role': 'driver', 'mode': 'm1', 'replicate': False},
            'driver_2': {'role': 'driver', 'mode': 'm2', 'replicate': False},
            'driver_3': {'role': 'driver', 'mode': 'm3', 'replicate': False},
            'driver_4': {'role': 'driver', 'mode': 'm3', 'replicate': False},
            'nested_subroutine_3': {'ignore': ['test1', 'test2']}
        },
        'transformations': {
            'Idem1': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
            'Idem2': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
            'Idem3': {'classname': 'IdemTransformation', 'module': 'loki.transformations'}
        },
        'pipelines': {
            # NB: The pipeline name is important here since it determines the mode of the items
            #     that don't have an explicit mode set in the config (e.g. nested_subroutine_3)
            'm1': {'transformations': {'Idem1'}},
            'm2': {'transformations': {'Idem2'}},
            'm3': {'transformations': {'Idem3'}}
        },
    }


@pytest.fixture(name='check_multimodes_callgraph', scope='module')
def fixture_check_multimodes_callgraph():
    """
    Fixture that provides a function to check that the call graph of the scheduler
    is consistent with the provided callgraph dictionary
    """
    def check_callgraph(scheduler, callgraph):
        if not isinstance(callgraph, dict) or not callgraph:
            return
        for routine in callgraph.keys():
            routine_ir = scheduler[routine].ir
            imports = routine_ir.imports
            imported_symbols = ()
            for imp in imports:
                imported_symbols += imp.symbols
            successors = []
            for successor in callgraph[routine]:
                if '#' in successor:
                    successors.append(scheduler[successor].ir)
            successors_local_name = [str(successor.name).lower() for successor in successors]
            calls = FindNodes(ir.CallStatement).visit(routine_ir.body)
            for call in calls:
                assert str(call.name).lower() in successors_local_name
                assert str(call.name).lower() in imported_symbols
            check_callgraph(scheduler, callgraph[routine])
    yield check_callgraph


def test_scheduler_multi_modes_invalid_pipeline(testdir, tmp_path, multi_modes_config):
    """
    Test that the scheduler raises an error when a pipeline is missing for a mode
    """
    config = SchedulerConfig.from_dict(multi_modes_config)
    config.pipelines.pop('m1')

    proj_multi_mode = testdir/'sources/projMultiMode'
    builddir = tmp_path/'scheduler_multi_driver_modes_dir'
    builddir.mkdir(exist_ok=True)

    scheduler = Scheduler(paths=proj_multi_mode, config=config, xmods=[tmp_path], output_dir=builddir)
    with pytest.raises(RuntimeError):
        scheduler.process(config.pipelines, proc_strategy=ProcessingStrategy.PLAN)


@pytest.mark.parametrize('as_modules', [False, True])
@pytest.mark.parametrize('reinit_scheduler', [True, False])
def test_scheduler_multi_modes(testdir, tmp_path, multi_modes_config, reinit_scheduler,
                               as_modules, check_multimodes_callgraph):
    config = SchedulerConfig.from_dict(multi_modes_config)

    if as_modules:
        proj_multi_mode = testdir/'sources/projMultiModeModules'
    else:
        proj_multi_mode = testdir/'sources/projMultiMode'

    builddir = tmp_path/'scheduler_multi_driver_modes_dir'
    builddir.mkdir(exist_ok=True)

    scheduler = Scheduler(paths=proj_multi_mode, config=config, xmods=[tmp_path], output_dir=builddir)

    # We manually invoke the mode propagation to check that the correct modes are assigned
    scheduler._propagate_modes()
    expected_modes_per_item = {
        'driver_0_mod#driver_0': {'m1'},
        'driver_1_mod#driver_1': {'m1'},
        'driver_2_mod#driver_2': {'m2'},
        'driver_3_mod#driver_3': {'m3'},
        'driver_4_mod#driver_4': {'m3'},
        'subroutine_1_mod#subroutine_1': {'m1'},
        'subroutine_2_mod#subroutine_2': {'m2'},
        'subroutine_3_mod#subroutine_3': {'m1', 'm3'},
        'nested_subroutine_1_mod#nested_subroutine_1': {'m1', 'm3'},
        'nested_subroutine_2_mod#nested_subroutine_2': {'m2'},
        'nested_subroutine_3_mod#nested_subroutine_3': {'m1', 'm2', 'm3'}
    }

    if not as_modules:
        # strip scope prefixes for non-module items
        expected_modes_per_item = {
            (item[item.find('#'):] if 'driver' not in item else item): modes
            for item, modes in expected_modes_per_item.items()
        }

    assert {
        item.name: {item.mode} if 'driver' in item.name else item.trafo_data['inherited_mode']
        for item in scheduler.items
    } == expected_modes_per_item

    # Manually invoke the pipeline separation several times to ensure it is idempotent
    scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.PLAN)
    scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.PLAN)
    scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.PLAN)

    scheduler.process(config.pipelines, proc_strategy=ProcessingStrategy.PLAN)

    expected_items_per_mode = {
        'm1': {
            'driver_0_mod#driver_0',
            'driver_1_mod#driver_1',
            'subroutine_1_loki_m1_mod#subroutine_1_loki_m1',
            'subroutine_3_loki_m1_mod#subroutine_3_loki_m1',
            'nested_subroutine_1_loki_m1_mod#nested_subroutine_1_loki_m1',
            'nested_subroutine_3_loki_m1_mod#nested_subroutine_3_loki_m1',
        },
        'm2': {
            'driver_2_mod#driver_2',
            'subroutine_2_loki_m2_mod#subroutine_2_loki_m2',
            'nested_subroutine_2_loki_m2_mod#nested_subroutine_2_loki_m2',
            'nested_subroutine_3_loki_m2_mod#nested_subroutine_3_loki_m2',
        },
        'm3': {
            'driver_3_mod#driver_3',
            'driver_4_mod#driver_4',
            'subroutine_3_loki_m3_mod#subroutine_3_loki_m3',
            'nested_subroutine_1_loki_m3_mod#nested_subroutine_1_loki_m3',
            'nested_subroutine_3_loki_m3_mod#nested_subroutine_3_loki_m3',
        }
    }
    if not as_modules:
        expected_items_per_mode = {
            mod: {item[item.find('#'):] if 'driver' not in item else item for item in items}
            for mod, items in expected_items_per_mode.items()
        }

    items_per_mode = defaultdict(set)
    for item in scheduler.items:
        items_per_mode[item.mode].add(item.name)

    assert set(items_per_mode) == set(expected_items_per_mode)
    for _mode, _val in items_per_mode.items():
        assert expected_items_per_mode[_mode] == _val

    transformations = (
        ModuleWrapTransformation(module_suffix='_mod'),
        DependencyTransformation(suffix='_test', module_suffix='_mod'),
        FileWriteTransformation()
    )
    for transformation in transformations:
        scheduler.process(transformation, proc_strategy=ProcessingStrategy.PLAN)

    plan_trafo = CMakePlanTransformation(rootpath=proj_multi_mode)
    scheduler.process(transformation=plan_trafo, proc_strategy=ProcessingStrategy.PLAN)
    planfile = tmp_path/'planfile'
    plan_trafo.write_plan(planfile)

    # Validate the plan file content
    loki_plan = planfile.read_text()
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    expected_keys = {'LOKI_SOURCES_TO_TRANSFORM', 'LOKI_SOURCES_TO_APPEND', 'LOKI_SOURCES_TO_REMOVE'}
    assert set(plan_dict.keys()) == expected_keys

    _expected_files_to_transform = {
        'driver_0_mod', 'driver_1_mod', 'driver_2_mod', 'driver_3_mod', 'driver_4_mod',
        'nested_subroutine_1_mod', 'nested_subroutine_2_mod', 'nested_subroutine_3_mod',
        'subroutine_1_mod', 'subroutine_2_mod', 'subroutine_3_mod'
    }
    if as_modules:
        expected_files_to_transform = _expected_files_to_transform
    else:
        expected_files_to_transform = {v.replace('_mod', '') if 'routine' in v else v
                                       for v in _expected_files_to_transform}
    _expected_files_to_append = {
        'driver_0_mod.m1', 'driver_1_mod.m1', 'driver_2_mod.m2', 'driver_3_mod.m3',
        'driver_4_mod.m3', 'nested_subroutine_1_loki_m1_mod.m1',
        'nested_subroutine_1_loki_m3_mod.m3', 'nested_subroutine_2_loki_m2_mod.m2',
        'nested_subroutine_3_loki_m1_mod.m1',
        'nested_subroutine_3_loki_m2_mod.m2', 'nested_subroutine_3_loki_m3_mod.m3',
        'subroutine_1_loki_m1_mod.m1', 'subroutine_2_loki_m2_mod.m2',
        'subroutine_3_loki_m1_mod.m1', 'subroutine_3_loki_m3_mod.m3'
    }
    if as_modules:
        expected_files_to_append = _expected_files_to_append
    else:
        expected_files_to_append = {v.replace('_mod', '') if 'routine' in v else v
                                    for v in _expected_files_to_append}
    expected_files_to_remove = {
        'driver_0_mod', 'driver_1_mod', 'driver_2_mod', 'driver_3_mod', 'driver_4_mod'
    }

    assert set(plan_dict['LOKI_SOURCES_TO_TRANSFORM']) == expected_files_to_transform
    assert set(plan_dict['LOKI_SOURCES_TO_APPEND']) == expected_files_to_append
    assert set(plan_dict['LOKI_SOURCES_TO_REMOVE']) == expected_files_to_remove

    if reinit_scheduler:
        scheduler = Scheduler(paths=proj_multi_mode, config=config, xmods=[tmp_path], output_dir=builddir)
    scheduler.propagate_and_separate_modes()
    for transformation in transformations:
        scheduler.process(transformation)

    expected_callgraph = {
        'driver_0_mod#driver_0': {
            'subroutine_1_loki_m1_test_mod#subroutine_1_loki_m1_test': {
                'nested_subroutine_1_loki_m1_test_mod#nested_subroutine_1_loki_m1_test',
            },
            'subroutine_3_loki_m1_test_mod#subroutine_3_loki_m1_test': {
                'nested_subroutine_1_loki_m1_test_mod#nested_subroutine_1_loki_m1_test',
                'nested_subroutine_3_loki_m1_test_mod#nested_subroutine_3_loki_m1_test',
            },
        },
        'driver_1_mod#driver_1': {
            'subroutine_1_loki_m1_test_mod#subroutine_1_loki_m1_test': {
                'nested_subroutine_1_loki_m1_test_mod#nested_subroutine_1_loki_m1_test',
            },
            'subroutine_3_loki_m1_test_mod#subroutine_3_loki_m1_test': {
                'nested_subroutine_1_loki_m1_test_mod#nested_subroutine_1_loki_m1_test',
                'nested_subroutine_3_loki_m1_test_mod#nested_subroutine_3_loki_m1_test',
            }
        },
        'driver_2_mod#driver_2': {
            'subroutine_2_loki_m2_test_mod#subroutine_2_loki_m2_test': {
                'nested_subroutine_2_loki_m2_test_mod#nested_subroutine_2_loki_m2_test',
                'nested_subroutine_3_loki_m2_test_mod#nested_subroutine_3_loki_m2_test'
            }
        },
        'driver_3_mod#driver_3': {
            'subroutine_3_loki_m3_test_mod#subroutine_3_loki_m3_test': {
                'nested_subroutine_1_loki_m3_test_mod#nested_subroutine_1_loki_m3_test',
                'nested_subroutine_3_loki_m3_test_mod#nested_subroutine_3_loki_m3_test',
            }
        },
        'driver_4_mod#driver_4': {
            'subroutine_3_loki_m3_test_mod#subroutine_3_loki_m3_test': {
                'nested_subroutine_1_loki_m3_test_mod#nested_subroutine_1_loki_m3_test',
                'nested_subroutine_3_loki_m3_test_mod#nested_subroutine_3_loki_m3_test',
            }
        }
    }

    check_multimodes_callgraph(scheduler, expected_callgraph)


@pytest.mark.parametrize('proc_strategy', [ProcessingStrategy.PLAN, ProcessingStrategy.DEFAULT])
def test_scheduler_multi_modes_imports(tmp_path, proc_strategy, check_multimodes_callgraph):
    """
    Test that the correct imports are added to the transformed files in multi-mode pipelines
    """

    fcode_driver1 = """
subroutine driver1
    use test_multi_modes_kernel1, only: kernel1, n
    use test_multi_modes_kernel2, only: kernel2
    implicit none
    integer :: arr(n, n)
    call kernel1
    call kernel2
end subroutine driver1
    """.strip()

    fcode_driver2 = """
subroutine driver2
    use test_multi_modes_kernel1, only: n
    use test_multi_modes_kernel2, only: kernel2
    implicit none
    integer :: arr(n, n)
    call kernel2
end subroutine driver2
    """.strip()

    fcode_kernel1 = """
module test_multi_modes_kernel1
implicit none
integer, parameter :: n = 10
contains
subroutine kernel1
    use test_multi_modes_kernel2, only: kernel2
    call kernel2
end subroutine kernel1
end module test_multi_modes_kernel1
    """.strip()

    fcode_kernel2 = """
module test_multi_modes_kernel2
implicit none
contains
subroutine kernel2
end subroutine kernel2
end module test_multi_modes_kernel2
    """.strip()

    config = {
        'default': {'role': 'kernel', 'expand': True, 'strict': False, 'mode': 'm1', 'replicate': True},
        'routines': {
            'driver1': {'role': 'driver', 'mode': 'm1', 'replicate': False},
            'driver2': {'role': 'driver', 'mode': 'm2', 'replicate': False},
        },
        'transformations': {
            'Idem1': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
            'Idem2': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
        },
        'pipelines': {
            'm1': {'transformations': {'Idem1'}},
            'm2': {'transformations': {'Idem2'}},
        },
    }
    scheduler_config = SchedulerConfig.from_dict(config)

    src_dir = tmp_path/'sources'
    src_dir.mkdir()
    (src_dir/'driver1.F90').write_text(fcode_driver1)
    (src_dir/'driver2.F90').write_text(fcode_driver2)
    (src_dir/'test_multi_modes_kernel1.F90').write_text(fcode_kernel1)
    (src_dir/'test_multi_modes_kernel2.F90').write_text(fcode_kernel2)

    out_dir = tmp_path/'output'
    out_dir.mkdir()

    scheduler = Scheduler(paths=src_dir, config=scheduler_config, xmods=[tmp_path], output_dir=out_dir)
    scheduler.propagate_and_separate_modes(proc_strategy=proc_strategy)
    scheduler.process(scheduler_config.pipelines, proc_strategy=proc_strategy)

    if proc_strategy != ProcessingStrategy.PLAN:
        expected_dependencies = {
            '#driver1': {
                'test_multi_modes_kernel1', 'test_multi_modes_kernel1_loki_m1#kernel1_loki_m1',
                'test_multi_modes_kernel2_loki_m1#kernel2_loki_m1'
            },
            '#driver2': {'test_multi_modes_kernel1', 'test_multi_modes_kernel2_loki_m2#kernel2_loki_m2'},
            'test_multi_modes_kernel1_loki_m1#kernel1_loki_m1': {'test_multi_modes_kernel2_loki_m1#kernel2_loki_m1'},
        }

        check_multimodes_callgraph(scheduler, expected_dependencies)

        expected_imports = {
            '#driver1': {
                'test_multi_modes_kernel1', 'test_multi_modes_kernel1_loki_m1', 'test_multi_modes_kernel2_loki_m1'
            },
            '#driver2': {'test_multi_modes_kernel1', 'test_multi_modes_kernel2_loki_m2'},
            'test_multi_modes_kernel1_loki_m1#kernel1_loki_m1': {'test_multi_modes_kernel2_loki_m1'},
        }
        for item_name, imports in expected_imports.items():
            item_ir = scheduler[item_name].ir
            item_imports = item_ir.imports
            imported_modules = {imp.module for imp in item_imports}
            assert imported_modules == imports


def _write_shared_kernel_multi_mode_sources(src_dir):
    """
    Create a minimal multi-mode graph with two drivers sharing one kernel that
    calls a downstream leaf.
    """
    fcode_driver_small = """
subroutine driver_small
    use shared_kernel_mod, only: shared_kernel
    implicit none
    call shared_kernel
end subroutine driver_small
    """.strip()

    fcode_driver_stack = """
subroutine driver_stack
    use shared_kernel_mod, only: shared_kernel
    implicit none
    call shared_kernel
end subroutine driver_stack
    """.strip()

    fcode_shared_kernel = """
module shared_kernel_mod
implicit none
contains
subroutine shared_kernel
    use leaf_mod, only: leaf
    call leaf
end subroutine shared_kernel
end module shared_kernel_mod
    """.strip()

    fcode_leaf = """
module leaf_mod
implicit none
contains
subroutine leaf
end subroutine leaf
end module leaf_mod
    """.strip()

    (src_dir/'driver_small.F90').write_text(fcode_driver_small)
    (src_dir/'driver_stack.F90').write_text(fcode_driver_stack)
    (src_dir/'shared_kernel_mod.F90').write_text(fcode_shared_kernel)
    (src_dir/'leaf_mod.F90').write_text(fcode_leaf)


def _shared_kernel_multi_mode_config():
    return SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': False, 'mode': 'm_small', 'replicate': True},
        'routines': {
            'driver_small': {'role': 'driver', 'mode': 'm_small', 'replicate': False},
            'driver_stack': {'role': 'driver', 'mode': 'm_stack', 'replicate': False},
        },
        'transformations': {
            'IdemSmall': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
            'IdemStack': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
        },
        'pipelines': {
            'm_small': {'transformations': {'IdemSmall'}},
            'm_stack': {'transformations': {'IdemStack'}},
        },
    })


def _shared_kernel_multi_mode_override_config():
    return SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': False, 'mode': 'm_small', 'replicate': True},
        'routines': {
            'driver_small': {'role': 'driver', 'mode': 'm_small', 'replicate': False},
            'driver_stack': {'role': 'driver', 'mode': 'm_stack', 'replicate': False},
            'shared_kernel': {'role': 'kernel'},
            'leaf': {'role': 'kernel'},
        },
        'transformations': {
            'IdemSmall': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
            'IdemStack': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
        },
        'pipelines': {
            'm_small': {'transformations': {'IdemSmall'}},
            'm_stack': {'transformations': {'IdemStack'}},
        },
    })


def _shared_kernel_multi_mode_seed_override_config():
    return SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': False, 'mode': 'm_small', 'replicate': True},
        'routines': {
            'driver_small': {'role': 'driver', 'mode': 'm_small', 'replicate': False},
            'driver_stack': {'role': 'driver', 'mode': 'm_stack', 'replicate': False},
            'shared_kernel': {'role': 'kernel', 'seed_routine': True},
            'leaf': {'role': 'kernel', 'seed_routine': True},
        },
        'transformations': {
            'IdemSmall': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
            'IdemStack': {'classname': 'IdemTransformation', 'module': 'loki.transformations'},
        },
        'pipelines': {
            'm_small': {'transformations': {'IdemSmall'}},
            'm_stack': {'transformations': {'IdemStack'}},
        },
    })


def test_scheduler_multi_modes_shared_kernel_plan_variants(tmp_path):
    """
    Document current multi-mode planning behaviour for a shared kernel.

    Without explicit per-routine config entries, the original shared kernel is
    replaced entirely by renamed per-mode variants.
    """
    scheduler_config = _shared_kernel_multi_mode_config()

    src_dir = tmp_path/'sources'
    src_dir.mkdir()
    _write_shared_kernel_multi_mode_sources(src_dir)

    out_dir = tmp_path/'output'
    out_dir.mkdir()

    scheduler = Scheduler(paths=src_dir, config=scheduler_config, xmods=[tmp_path], output_dir=out_dir)
    scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.PLAN)
    scheduler.process(scheduler_config.pipelines, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.process(FileWriteTransformation(), proc_strategy=ProcessingStrategy.PLAN)

    plan_trafo = CMakePlanTransformation(rootpath=src_dir)
    scheduler.process(transformation=plan_trafo, proc_strategy=ProcessingStrategy.PLAN)
    planfile = tmp_path/'planfile'
    plan_trafo.write_plan(planfile)

    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(planfile.read_text())}
    append_stems = [Path(s).stem for s in plan_dict['LOKI_SOURCES_TO_APPEND']]

    assert len(plan_dict['LOKI_SOURCES_TO_APPEND']) == len(set(plan_dict['LOKI_SOURCES_TO_APPEND']))

    assert 'shared_kernel_mod.m_small' not in append_stems
    assert append_stems.count('shared_kernel_loki_m_small_mod.m_small') == 1
    assert append_stems.count('shared_kernel_loki_m_stack_mod.m_stack') == 1

    assert 'leaf_mod.m_small' not in append_stems
    assert append_stems.count('leaf_loki_m_small_mod.m_small') == 1
    assert append_stems.count('leaf_loki_m_stack_mod.m_stack') == 1


def test_scheduler_multi_modes_shared_kernel_plan_variants_with_routine_override(tmp_path):
    """
    Explicit per-routine config entries change the current behaviour: the
    original shared kernel remains and same-mode clones are emitted in addition.
    """
    scheduler_config = _shared_kernel_multi_mode_override_config()

    src_dir = tmp_path/'sources'
    src_dir.mkdir()
    _write_shared_kernel_multi_mode_sources(src_dir)

    out_dir = tmp_path/'output'
    out_dir.mkdir()

    scheduler = Scheduler(
        paths=src_dir, config=scheduler_config,
        seed_routines=['driver_small', 'driver_stack', 'shared_kernel', 'leaf'],
        xmods=[tmp_path], output_dir=out_dir
    )
    scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.PLAN)
    scheduler.process(scheduler_config.pipelines, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.process(FileWriteTransformation(), proc_strategy=ProcessingStrategy.PLAN)

    plan_trafo = CMakePlanTransformation(rootpath=src_dir)
    scheduler.process(transformation=plan_trafo, proc_strategy=ProcessingStrategy.PLAN)
    planfile = tmp_path/'planfile'
    plan_trafo.write_plan(planfile)

    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(planfile.read_text())}
    append_stems = [Path(s).stem for s in plan_dict['LOKI_SOURCES_TO_APPEND']]

    assert len(plan_dict['LOKI_SOURCES_TO_APPEND']) == len(set(plan_dict['LOKI_SOURCES_TO_APPEND']))

    # This matches the semantic duplication pattern seen in the real-world case.
    assert append_stems.count('shared_kernel_mod.m_small') == 1
    assert append_stems.count('shared_kernel_loki_m_small_mod.m_small') == 1
    assert append_stems.count('shared_kernel_loki_m_stack_mod.m_stack') == 1

    assert append_stems.count('leaf_mod.m_small') == 1
    assert append_stems.count('leaf_loki_m_small_mod.m_small') == 1
    assert append_stems.count('leaf_loki_m_stack_mod.m_stack') == 1


def test_scheduler_multi_modes_shared_kernel_call_import_rewrite_consistency(tmp_path):
    """
    Document how call and import rewrites currently behave for shared kernels.

    Both drivers are rewired to mode-specific clones, while the original shared
    kernel and leaf remain in the scheduler with their original call/import chain.
    """
    scheduler_config = _shared_kernel_multi_mode_override_config()

    src_dir = tmp_path/'sources'
    src_dir.mkdir()
    _write_shared_kernel_multi_mode_sources(src_dir)

    out_dir = tmp_path/'output'
    out_dir.mkdir()

    scheduler = Scheduler(
        paths=src_dir, config=scheduler_config,
        seed_routines=['driver_small', 'driver_stack', 'shared_kernel', 'leaf'],
        xmods=[tmp_path], output_dir=out_dir
    )
    scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.DEFAULT)
    scheduler.process(scheduler_config.pipelines, proc_strategy=ProcessingStrategy.DEFAULT)

    expected_items = {
        '#driver_small',
        '#driver_stack',
        'shared_kernel_mod#shared_kernel',
        'shared_kernel_loki_m_small_mod#shared_kernel_loki_m_small',
        'shared_kernel_loki_m_stack_mod#shared_kernel_loki_m_stack',
        'leaf_mod#leaf',
        'leaf_loki_m_small_mod#leaf_loki_m_small',
        'leaf_loki_m_stack_mod#leaf_loki_m_stack',
    }
    assert expected_items <= {item.name for item in scheduler.items}

    def calls_for(item_name):
        routine = scheduler[item_name].ir
        return tuple(str(call.name).lower() for call in FindNodes(ir.CallStatement).visit(routine.body))

    def imports_for(item_name):
        routine = scheduler[item_name].ir
        return {imp.module.lower() for imp in routine.imports}

    assert calls_for('#driver_small') == ('shared_kernel_loki_m_small',)
    assert imports_for('#driver_small') == {'shared_kernel_loki_m_small_mod'}

    assert calls_for('#driver_stack') == ('shared_kernel_loki_m_stack',)
    assert imports_for('#driver_stack') == {'shared_kernel_loki_m_stack_mod'}

    assert calls_for('shared_kernel_loki_m_small_mod#shared_kernel_loki_m_small') == ('leaf_loki_m_small',)
    assert imports_for('shared_kernel_loki_m_small_mod#shared_kernel_loki_m_small') == {'leaf_loki_m_small_mod'}

    assert calls_for('shared_kernel_loki_m_stack_mod#shared_kernel_loki_m_stack') == ('leaf_loki_m_stack',)
    assert imports_for('shared_kernel_loki_m_stack_mod#shared_kernel_loki_m_stack') == {'leaf_loki_m_stack_mod'}

    # The original shared kernel remains active with the original import/call chain.
    assert calls_for('shared_kernel_mod#shared_kernel') == ('leaf',)
    assert imports_for('shared_kernel_mod#shared_kernel') == {'leaf_mod'}


def test_scheduler_multi_modes_explicit_routine_entries_do_not_become_seeds(tmp_path):
    """
    Explicit routine config entries alone do not become scheduler seeds when
    seed_routines is not provided.
    """
    src_dir = tmp_path/'sources'
    src_dir.mkdir()
    _write_shared_kernel_multi_mode_sources(src_dir)

    no_override_scheduler = Scheduler(
        paths=src_dir, config=_shared_kernel_multi_mode_config(),
        xmods=[tmp_path], output_dir=tmp_path/'out-no-override'
    )
    override_scheduler = Scheduler(
        paths=src_dir, config=_shared_kernel_multi_mode_override_config(), xmods=[tmp_path],
        output_dir=tmp_path/'out-override'
    )

    assert no_override_scheduler.seeds == ('driver_small', 'driver_stack')
    assert override_scheduler.seeds == ('driver_small', 'driver_stack')

    no_override_scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.DEFAULT)
    override_scheduler.propagate_and_separate_modes(proc_strategy=ProcessingStrategy.DEFAULT)

    assert 'shared_kernel_mod#shared_kernel' not in {item.name for item in no_override_scheduler.items}
    assert 'shared_kernel_mod#shared_kernel' not in {item.name for item in override_scheduler.items}


def test_scheduler_implicit_seeds_include_seed_routine_entries(tmp_path):
    """
    Non-driver routines explicitly marked with ``seed_routine = true`` are added
    to the implicit seed set alongside driver routines.
    """
    src_dir = tmp_path/'sources'
    src_dir.mkdir()
    _write_shared_kernel_multi_mode_sources(src_dir)

    scheduler = Scheduler(
        paths=src_dir, config=_shared_kernel_multi_mode_seed_override_config(),
        xmods=[tmp_path], output_dir=tmp_path/'output'
    )

    assert scheduler.seeds == ('driver_small', 'driver_stack', 'shared_kernel', 'leaf')


def test_scheduler_requires_implicit_seed_or_explicit_seed_routines(tmp_path, caplog):
    """
    When no explicit seeds are provided, the scheduler requires at least one
    implicit seed via a driver role or ``seed_routine = true``.
    """
    src_dir = tmp_path/'sources'
    src_dir.mkdir()
    _write_shared_kernel_multi_mode_sources(src_dir)

    config = SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': False},
        'routines': {
            'shared_kernel': {'role': 'kernel'},
            'leaf': {'role': 'kernel'},
        },
    })

    caplog.set_level(WARNING)

    # Once deprecated:
    # with pytest.raises(RuntimeError, match='No seed routines provided and no implicit seeds found'):
    scheduler = Scheduler(paths=src_dir, config=config, xmods=[tmp_path], output_dir=tmp_path/'output')
    # however, rn for backwards compability only expect warning and old behaviour
    assert set(scheduler.seeds) == {'shared_kernel', 'leaf'}
    assert len(caplog.records) == 1
    assert "Using implicit or unspecified seed routines is deprecated and will" in caplog.records[0].message
