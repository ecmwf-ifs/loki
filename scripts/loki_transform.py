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
    Frontend, as_tuple, set_excepthook, auto_post_mortem_debugger, info,
    GlobalVarImportItem, Module
)

# Get generalized transformations provided by Loki
from loki.transform import (
    DependencyTransformation, ModuleWrapTransformation, FortranCTransformation,
    FileWriteTransformation, HoistTemporaryArraysAnalysis, normalize_range_indexing,
    InlineTransformation, SanitiseTransformation
)

# pylint: disable=wrong-import-order
from transformations.argument_shape import ArgumentArrayShapeAnalysis, ExplicitArgumentArrayShapeTransformation
from transformations.data_offload import (
    DataOffloadTransformation, GlobalVariableAnalysis, GlobalVarOffloadTransformation
)
from transformations.derived_types import DerivedTypeArgumentsTransformation
from transformations.utility_routines import DrHookTransformation, RemoveCallsTransformation
from transformations.pool_allocator import TemporariesPoolAllocatorTransformation
from transformations.single_column_claw import ExtractSCATransformation, CLAWTransformation
from transformations.single_column_coalesced import (
    SCCBaseTransformation, SCCAnnotateTransformation,
    SCCHoistTemporaryArraysTransformation
)
from transformations.single_column_coalesced_vector import (
    SCCDevectorTransformation, SCCRevectorTransformation, SCCDemoteTransformation
)
from transformations.scc_cuf import (
    SccCufTransformation, SccCufTransformationNew, HoistTemporaryArraysDeviceAllocatableTransformation
)
from transformations.single_column_coalesced_extended import (
        SCCLowerLoopTransformation
)

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
                  ['idem', 'c', 'scc-cpu', 'idem-stack', 'sca', 'claw', 'scc', 'scc-hoist', 'scc-stack',
                   'cuf-parametrise', 'cuf-parametrise-new', 'cuf-hoist', 'cuf-hoist-new', 'cuf-dynamic']
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
@click.option('--assume-deviceptr', is_flag=True, default=False,
              help='Mark the relevant arguments as true device-pointers in "!$loki data" regions.')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--trim-vector-sections', is_flag=True, default=False,
              help='Trim vector loops in SCC transform to exclude scalar assignments.')
@click.option('--global-var-offload', is_flag=True, default=False,
              help="Generate offload instructions for global vars imported via 'USE' statements.")
@click.option('--remove-derived-args/--no-remove-derived-args', default=False,
              help="Remove derived-type arguments and replace with canonical arguments")
@click.option('--inline-members/--no-inline-members', default=False,
              help='Inline member functions for SCC-class transformations.')
@click.option('--inline-marked/--no-inline-marked', default=True,
              help='Inline pragma-marked subroutines for SCC-class transformations.')
@click.option('--resolve-sequence-association/--no-resolve-sequence-association', default=False,
              help='Replace array arguments passed as scalars with arrays.')
@click.option('--derive-argument-array-shape/--no-derive-argument-array-shape', default=False,
              help="Recursively derive explicit shape dimension for argument arrays")
@click.option('--eliminate-dead-code/--no-eliminate-dead-code', default=True,
              help='Perform dead code elimination, where unreachable branches are trimmed from the code.')
def convert(
        mode, config, build, source, header, cpp, directive, include, define, omni_include, xmod,
        data_offload, remove_openmp, assume_deviceptr, frontend, trim_vector_sections,
        global_var_offload, remove_derived_args, inline_members, inline_marked,
        resolve_sequence_association, derive_argument_array_shape, eliminate_dead_code
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
    # definitions with new scheduler not necessary anymore. However, "source" need to be adjusted 
    #  in order to allow the scheduler to find the dependencies
    definitions = []
    for h in header:
        sfile = Sourcefile.from_file(filename=h, frontend=frontend_type, definitions=definitions,
                **build_args)
        definitions = definitions + list(sfile.definitions)

    # Create a scheduler to bulk-apply source transformations
    paths = [Path(p).resolve() for p in as_tuple(source)]
    paths += [Path(h).resolve().parent for h in as_tuple(header)]
    scheduler = Scheduler(
        paths=paths, config=config, frontend=frontend, definitions=definitions, **build_args
    )

    # Pull dimension definition from configuration
    horizontal = scheduler.config.dimensions.get('horizontal', None)
    vertical = scheduler.config.dimensions.get('vertical', None)
    block_dim = scheduler.config.dimensions.get('block_dim', None)

    # First, remove all derived-type arguments; caller first!
    if remove_derived_args:
        scheduler.process( DerivedTypeArgumentsTransformation() )

    # Remove DR_HOOK and other utility calls first, so they don't interfere with SCC loop hoisting
    if 'scc' in mode or mode == 'cuf-parametrise-new':
        scheduler.process( RemoveCallsTransformation(
            routines=config.default.get('utility_routines', None) or ['DR_HOOK', 'ABOR1', 'WRITE(NULOUT'],
            include_intrinsics=True, kernel_only=True
        ))
    else:
        scheduler.process( DrHookTransformation(mode=mode, remove=False) )

    # Perform general source sanitisation steps to level the playing field
    sanitise_trafo = scheduler.config.transformations.get('SanitiseTransformation', None)
    if not sanitise_trafo:
        sanitise_trafo = SanitiseTransformation(
            resolve_sequence_association=resolve_sequence_association,
        )
    scheduler.process(transformation=sanitise_trafo)

    # Perform source-inlining either from CLI arguments or from config
    inline_trafo = scheduler.config.transformations.get('InlineTransformation', None)
    if not inline_trafo:
        inline_trafo = InlineTransformation(
            inline_internals=inline_members, inline_marked=inline_marked,
            eliminate_dead_code=eliminate_dead_code, allowed_aliases=horizontal.index
        )
    scheduler.process(transformation=inline_trafo)

    # Backward insert argument shapes (for surface routines)
    if derive_argument_array_shape:
        scheduler.process(transformation=ArgumentArrayShapeAnalysis())
        scheduler.process(transformation=ExplicitArgumentArrayShapeTransformation())

    # Insert data offload regions for GPUs and remove OpenMP threading directives
    use_claw_offload = True
    if data_offload:
        offload_transform = DataOffloadTransformation(
            remove_openmp=remove_openmp, assume_deviceptr=assume_deviceptr
        )
        scheduler.process(offload_transform)
        use_claw_offload = not offload_transform.has_data_regions

    # Now we instantiate our transformation pipeline and apply the main changes
    transformation = None
    if mode in ['idem', 'idem-stack']:
        scheduler.process( IdemTransformation() )

    if mode == 'sca':
        scheduler.process( ExtractSCATransformation(horizontal=horizontal) )

    if mode == 'claw':
        scheduler.process( CLAWTransformation(
            horizontal=horizontal, claw_data_offload=use_claw_offload
        ))

    if mode in ['scc', 'scc-hoist', 'scc-stack', 'scc-cpu', 'cuf-parametrise-new']:
        # Apply the basic SCC transformation set
        scheduler.process( SCCBaseTransformation(
            horizontal=horizontal, directive=directive
        ))
        scheduler.process( SCCDevectorTransformation(
            horizontal=horizontal, trim_vector_sections=trim_vector_sections
        ))
        scheduler.process( SCCDemoteTransformation(horizontal=horizontal))
        scheduler.process( SCCRevectorTransformation(horizontal=horizontal))

    if mode == 'scc-hoist':
        # Apply recursive hoisting of local temporary arrays.
        # This requires a first analysis pass to run in reverse
        # direction through the call graph to gather temporary arrays.
        scheduler.process( HoistTemporaryArraysAnalysis(dim_vars=(vertical.size,)) )
        scheduler.process( SCCHoistTemporaryArraysTransformation(block_dim=block_dim) )

    if mode in ['scc', 'scc-hoist', 'scc-stack']:
        scheduler.process( SCCAnnotateTransformation(
                horizontal=horizontal, vertical=vertical, directive=directive, block_dim=block_dim
        ))
    
    if mode in ['scc-cpu']:
        scheduler.process(SCCLowerLoopTransformation(dimension=block_dim, dim_name='IBL'))

    if mode == "cuf-parametrise-new" or mode == "cuf-hoist-new":
        scc_extended = SCCLowerLoopTransformation(
                dimension=block_dim, dim_name='ibl', keep_driver_loop=True,
                ignore_dim_name=True
        )
        scheduler.process(transformation=scc_extended)

        # cuf_transform = SccCufTransformationNew(
        #     horizontal=horizontal, vertical=vertical, block_dim=block_dim,
        #     transformation_type='parametrise'
        # )
        # scheduler.process(transformation=cuf_transform)
        scheduler.process( transformation=scheduler.config.transformations[mode] )

    if mode in ['cuf-parametrise', 'cuf-hoist', 'cuf-dynamic']:
        # These transformations requires complex constructor arguments,
        # so we use the file-based transformation configuration.
        scheduler.process( transformation=scheduler.config.transformations[mode] )

    if global_var_offload:
        scheduler.process(transformation=GlobalVariableAnalysis())
        scheduler.process(transformation=GlobalVarOffloadTransformation())

    if mode in ['idem-stack', 'scc-stack']:
        if frontend == Frontend.OMNI:
            # To make the pool allocator size derivation work correctly, we need
            # to normalize the 1:end-style index ranges that OMNI introduces
            class NormalizeRangeIndexingTransformation(Transformation):
                def transform_subroutine(self, routine, **kwargs):
                    normalize_range_indexing(routine)

            scheduler.process( NormalizeRangeIndexingTransformation() )

        directive = {'idem-stack': 'openmp', 'scc-stack': 'openacc'}[mode]
        scheduler.process(transformation=TemporariesPoolAllocatorTransformation(
            block_dim=block_dim, directive=directive, check_bounds='scc' not in mode
        ))

    if mode == 'cuf-parametrise' or mode == "cuf-parametrise-new":
        # This transformation requires complex constructora arguments,
        # so we use the file-based transformation configuration.
        transformation = scheduler.config.transformations['ParametriseTransformation']
        scheduler.process(transformation=transformation)

    if mode == "cuf-hoist" or mode == "cuf-hoist-new":
        vertical = scheduler.config.dimensions['vertical']
        scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=(vertical.size,)))
        scheduler.process(transformation=HoistTemporaryArraysDeviceAllocatableTransformation(as_kwarguments=True))

    mode = mode.replace('-', '_')  # Sanitize mode string
    if mode in ["c"]:
        f2c_transformation = FortranCTransformation(path=build)
        scheduler.process(f2c_transformation)
        for h in definitions:
            f2c_transformation.apply(h, role='header')
        # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
        dependency = DependencyTransformation(suffix='_FC', module_suffix='_MOD')
        scheduler.process(dependency)
    else:
        # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
        scheduler.process( ModuleWrapTransformation(module_suffix='_MOD') )
        scheduler.process( DependencyTransformation(suffix=f'_{mode.upper()}', module_suffix='_MOD') )

    # Write out all modified source files into the build directory
    scheduler.process(transformation=FileWriteTransformation(
        builddir=build, mode=mode, cuf='cuf' in mode,
        include_module_var_imports=global_var_offload
    ))


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
