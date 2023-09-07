#!/usr/bin/env python

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Loki head script for source-to-source transformations concerning ECMWF
physics, including "Single Column" (SCA) and CLAW transformations.
"""

from pathlib import Path
import click

from loki import (
    Sourcefile, Transformation, Scheduler, SchedulerConfig, SubroutineItem,
    Frontend, as_tuple, set_excepthook, auto_post_mortem_debugger, flatten, info,
    GlobalVarImportItem
)

# Get generalized transformations provided by Loki
from loki.transform import (
    DependencyTransformation, FortranCTransformation, FileWriteTransformation,
    ParametriseTransformation, HoistTemporaryArraysAnalysis, normalize_range_indexing
)

# pylint: disable=wrong-import-order
from transformations.data_offload import DataOffloadTransformation, GlobalVarOffloadTransformation
from transformations.derived_types import DerivedTypeArgumentsTransformation
from transformations.utility_routines import DrHookTransformation, RemoveCallsTransformation
from transformations.pool_allocator import TemporariesPoolAllocatorTransformation
from transformations.single_column_claw import ExtractSCATransformation, CLAWTransformation
from transformations.single_column_coalesced import (
     SCCBaseTransformation, SCCAnnotateTransformation, SCCHoistTransformation
)
from transformations.single_column_coalesced_vector import (
     SCCDevectorTransformation, SCCRevectorTransformation, SCCDemoteTransformation
)
from transformations.scc_cuf import SccCufTransformation, HoistTemporaryArraysDeviceAllocatableTransformation


class IdemTransformation(Transformation):
    """
    A custom transformation pipeline that primarily does nothing,
    allowing us to test simple parse-unparse cycles.
    """

    def transform_subroutine(self, routine, **kwargs):
        pass


@click.group()
@click.option('--debug/--no-debug', default=False, show_default=True,
              help=('Enable / disable debug mode. This automatically attaches '
                    'a debugger when exceptions occur'))
def cli(debug):
    if debug:
        set_excepthook(hook=auto_post_mortem_debugger)


@cli.command()
@click.option('--mode', '-m', default='idem',
              type=click.Choice(
                  ['idem', 'idem-stack', 'sca', 'claw', 'scc', 'scc-hoist', 'scc-stack',
                   'cuf-parametrise', 'cuf-hoist', 'cuf-dynamic']
              ),
              help='Transformation mode, selecting which code transformations to apply.')
@click.option('--config', default=None, type=click.Path(),
              help='Path to custom scheduler configuration file')
@click.option('--build', '-b', '--out-path', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--source', '-s', '--path', type=click.Path(), multiple=True,
              help='Path to search during source exploration.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--directive', default='openacc', type=click.Choice(['openacc', 'openmp', 'none']),
              help='Programming model directives to insert (default openacc)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--define', '-D', multiple=True,
              help='Additional symbol definitions for the C-preprocessor')
@click.option('--omni-include', type=click.Path(), multiple=True,
              help='Additional path for header files, specifically for OMNI')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional .xmod file(s) for OMNI')
@click.option('--data-offload', is_flag=True, default=False,
              help='Run transformation to insert custom data offload regions.')
@click.option('--remove-openmp', is_flag=True, default=False,
              help='Removes existing OpenMP pragmas in "!$loki data" regions.')
@click.option('--deviceptr', is_flag=True, default=False,
              help='Mark the relevant arguments as true device-pointers in "!$loki data" regions.')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--trim-vector-sections', is_flag=True, default=False,
              help='Trim vector loops in SCC transform to exclude scalar assignments.')
@click.option('--global-var-offload', is_flag=True, default=False,
              help="Generate offload instructions for global vars imported via 'USE' statements.")
@click.option('--remove-derived-args/--no-remove-derived-args', default=False,
              help="Remove derived-type arguments and replace with canonical arguments")
def convert(
        mode, config, build, source, header, cpp, directive, include, define, omni_include, xmod,
        data_offload, remove_openmp, deviceptr, frontend, trim_vector_sections,
        global_var_offload, remove_derived_args
):
    """
    Batch-processing mode for Fortran-to-Fortran transformations that
    employs a :class:`Scheduler` to process large numbers of source
    files.

    Based on the given "mode" string, configuration file, source file
    paths and build arguments the :any:`Scheduler` will perform
    automatic call-tree exploration and apply a set of
    :any:`Transformation` objects to this call tree.
    """

    info(f'[Loki] Batch-processing source files using config: {config} ')

    config = SchedulerConfig.from_file(config)

    directive = None if directive.lower() == 'none' else directive.lower()

    build_args = {
        'preprocess': cpp,
        'includes': include,
        'defines': define,
        'xmods': xmod,
        'omni_includes': omni_include,
    }

    frontend = Frontend[frontend.upper()]
    frontend_type = Frontend.FP if frontend == Frontend.OMNI else frontend

    # Note, in order to get function inlinig correct, we need full knowledge
    # of any imported symbols and functions. Since we cannot yet retro-fit that
    # after creation, we need to make sure that the order of definitions can
    # be used to create a coherent stack of type definitions.
    definitions = []
    for h in header:
        sfile = Sourcefile.from_file(filename=h, frontend=frontend_type, **build_args)
        definitions = definitions + list(sfile.modules)

    # Create a scheduler to bulk-apply source transformations
    paths = [Path(p).resolve() for p in as_tuple(source)]
    paths += [Path(h).resolve().parent for h in as_tuple(header)]
    scheduler = Scheduler(
        paths=paths, config=config, frontend=frontend, definitions=definitions, **build_args
    )

    # First, remove all derived-type arguments; caller first!
    if remove_derived_args:
        scheduler.process(transformation=DerivedTypeArgumentsTransformation())

    # Remove DR_HOOK and other utility calls first, so they don't interfere with SCC loop hoisting
    if 'scc' in mode:
        scheduler.process(transformation=RemoveCallsTransformation(
            routines=config.default.get('utility_routines', None) or ['DR_HOOK', 'ABOR1', 'WRITE(NULOUT'],
            include_intrinsics=True
        ))
    else:
        scheduler.process(transformation=DrHookTransformation(mode=mode, remove=False))

    # Insert data offload regions for GPUs and remove OpenMP threading directives
    use_claw_offload = True
    if data_offload:
        offload_transform = DataOffloadTransformation(remove_openmp=remove_openmp, deviceptr=deviceptr)
        scheduler.process(transformation=offload_transform)
        use_claw_offload = not offload_transform.has_data_regions

    # Now we instantiate our transformation pipeline and apply the main changes
    transformation = None
    if mode in ['idem', 'idem-stack']:
        transformation = (IdemTransformation(),)

    if mode == 'sca':
        horizontal = scheduler.config.dimensions['horizontal']
        transformation = (ExtractSCATransformation(horizontal=horizontal),)

    if mode == 'claw':
        horizontal = scheduler.config.dimensions['horizontal']
        transformation = (CLAWTransformation(
            horizontal=horizontal, claw_data_offload=use_claw_offload
        ),)

    if mode in ['scc', 'scc-hoist', 'scc-stack']:
        horizontal = scheduler.config.dimensions['horizontal']
        vertical = scheduler.config.dimensions['vertical']
        block_dim = scheduler.config.dimensions['block_dim']
        transformation = (SCCBaseTransformation(horizontal=horizontal, directive=directive),)
        transformation += (SCCDevectorTransformation(horizontal=horizontal, trim_vector_sections=trim_vector_sections),)
        transformation += (SCCDemoteTransformation(horizontal=horizontal),)
        if not 'hoist' in mode:
            transformation += (SCCRevectorTransformation(horizontal=horizontal),)
        if 'hoist' in mode:
            transformation += (SCCHoistTransformation(horizontal=horizontal, vertical=vertical, block_dim=block_dim),)
        transformation += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical,
                                                     directive=directive, block_dim=block_dim,
                                                     hoist_column_arrays='hoist' in mode),)

    if mode in ['cuf-parametrise', 'cuf-hoist', 'cuf-dynamic']:
        horizontal = scheduler.config.dimensions['horizontal']
        vertical = scheduler.config.dimensions['vertical']
        block_dim = scheduler.config.dimensions['block_dim']
        derived_types = scheduler.config.derived_types
        transformation = (SccCufTransformation(
            horizontal=horizontal, vertical=vertical, block_dim=block_dim,
            transformation_type=mode.replace('cuf-', ''),
            derived_types=derived_types
        ),)

    if transformation:
        for transform in transformation:
            scheduler.process(transformation=transform)
    else:
        raise RuntimeError('[Loki] Convert could not find specified Transformation!')

    if global_var_offload:
        scheduler.process(transformation=GlobalVarOffloadTransformation(),
                          item_filter=(SubroutineItem, GlobalVarImportItem), reverse=True)

    if mode in ['idem-stack', 'scc-stack']:
        if frontend == Frontend.OMNI:
            # To make the pool allocator size derivation work correctly, we need
            # to normalize the 1:end-style index ranges that OMNI introduces
            class NormalizeRangeIndexingTransformation(Transformation):
                def transform_subroutine(self, routine, **kwargs):
                    normalize_range_indexing(routine)

            scheduler.process(transformation=NormalizeRangeIndexingTransformation())

        horizontal = scheduler.config.dimensions['horizontal']
        vertical = scheduler.config.dimensions['vertical']
        block_dim = scheduler.config.dimensions['block_dim']
        directive = {'idem-stack': 'openmp', 'scc-stack': 'openacc'}[mode]
        transformation = TemporariesPoolAllocatorTransformation(
            block_dim=block_dim, directive=directive, check_bounds='scc' not in mode
        )
        scheduler.process(transformation=transformation, reverse=True)
    if mode == 'cuf-parametrise':
        dic2p = scheduler.config.dic2p
        disable = scheduler.config.disable
        transformation = ParametriseTransformation(dic2p=dic2p, disable=disable)
        scheduler.process(transformation=transformation)
    if mode == "cuf-hoist":
        disable = scheduler.config.disable
        vertical = scheduler.config.dimensions['vertical']
        scheduler.process(transformation=HoistTemporaryArraysAnalysis(disable=disable, dim_vars=(vertical.size,)),
                          reverse=True)
        scheduler.process(transformation=HoistTemporaryArraysDeviceAllocatableTransformation(disable=disable))

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    mode = mode.replace('-', '_')  # Sanitize mode string
    dependency = DependencyTransformation(suffix=f'_{mode.upper()}',
                                          mode='module', module_suffix='_MOD')
    scheduler.process(transformation=dependency, use_file_graph=True)

    # Write out all modified source files into the build directory
    if global_var_offload:
        item_filter = (SubroutineItem, GlobalVarImportItem)
    else:
        item_filter = SubroutineItem
    scheduler.process(
        transformation=FileWriteTransformation(builddir=build, mode=mode, cuf='cuf' in mode),
        use_file_graph=True, item_filter=item_filter
    )


@cli.command()
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated souce files.')
@click.option('--header', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--define', '-I', multiple=True,
              help='Additional symbol definitions for C-preprocessor')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--frontend', default='omni', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
def transpile(out_path, header, source, driver, cpp, include, define, frontend, xmod):
    """
    Convert kernels to C and generate ISO-C bindings and interfaces.
    """
    driver_name = 'CLOUDSC_DRIVER'
    kernel_name = 'CLOUDSC'

    frontend = Frontend[frontend.upper()]
    frontend_type = Frontend.FP if frontend == Frontend.OMNI else frontend

    # Note, in order to get function inlinig correct, we need full knowledge
    # of any imported symbols and functions. Since we cannot yet retro-fit that
    # after creation, we need to make sure that the order of definitions can
    # be used to create a coherent stack of type definitions.
    definitions = []
    for h in header:
        sfile = Sourcefile.from_file(h, xmods=xmod, definitions=definitions,
                                     frontend=frontend_type, preprocess=cpp)
        definitions = definitions + list(sfile.definitions)

    # Parse original driver and kernel routine, and enrich the driver
    kernel = Sourcefile.from_file(source, definitions=definitions, preprocess=cpp,
                                  includes=include, defines=define, xmods=xmod,
                                  frontend=frontend)
    driver = Sourcefile.from_file(driver, xmods=xmod, frontend=frontend)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    kernel_item = SubroutineItem(f'#{kernel_name.lower()}', source=kernel)
    driver_item = SubroutineItem(f'#{driver_name.lower()}', source=driver)

    # First, remove all derived-type arguments; caller first!
    transformation = DerivedTypeArgumentsTransformation()
    kernel[kernel_name].apply(transformation, role='kernel', item=kernel_item)
    driver[driver_name].apply(transformation, role='driver', item=driver_item, successors=(kernel_item,))

    # Now we instantiate our pipeline and apply the changes
    transformation = FortranCTransformation()
    transformation.apply(kernel, role='kernel', path=out_path)

    # Traverse header modules to create getter functions for module variables
    for h in definitions:
        transformation.apply(h, role='header', path=out_path)

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_FC', mode='module', module_suffix='_MOD')
    kernel.apply(dependency, role='kernel', targets=())
    kernel.write(path=Path(out_path)/kernel.path.with_suffix('.c.F90').name)

    # Re-generate the driver that mimicks the original source file,
    # but imports and calls our re-generated kernel.
    driver.apply(dependency, role='driver', targets=kernel_name)
    driver.write(path=Path(out_path)/driver.path.with_suffix('.c.F90').name)


@cli.command('plan')
@click.option('--mode', '-m', default='sca',
              type=click.Choice(['idem', 'sca', 'claw', 'scc', 'scc-hoist']))
@click.option('--config', '-c', type=click.Path(),
              help='Path to configuration file.')
@click.option('--header', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--source', '-s', type=click.Path(), multiple=True,
              help='Path to source files to transform.')
@click.option('--build', '-b', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--root', type=click.Path(), default=None,
              help='Root path to which all paths are relative to.')
@click.option('--directive', default='openacc', type=click.Choice(['openacc', 'openmp', 'none']),
              help='Programming model directives to insert (default openacc)')
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--callgraph', '-g', type=click.Path(), default=None,
              help='Generate and display the subroutine callgraph.')
@click.option('--plan-file', type=click.Path(),
              help='CMake "plan" file to generate.')
def plan(mode, config, header, source, build, root, cpp, directive, frontend, callgraph, plan_file):
    """
    Create a "plan", a schedule of files to inject and transform for a
    given configuration.
    """

    info(f'[Loki] Creating CMake plan file from config: {config}')
    config = SchedulerConfig.from_file(config)

    paths = [Path(s).resolve() for s in source]
    paths += [Path(h).resolve().parent for h in header]
    scheduler = Scheduler(paths=paths, config=config, frontend=frontend, full_parse=False, preprocess=cpp)

    # Construct the transformation plan as a set of CMake lists of source files
    scheduler.write_cmake_plan(filepath=plan_file, mode=mode, buildpath=build, rootpath=root)

    # Output the resulting callgraph
    if callgraph:
        scheduler.callgraph(callgraph)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
