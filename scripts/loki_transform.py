#!/usr/bin/env python

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Loki head script for source-to-source transformations for batch processing
via the :any:`Scheduler`.
"""

from pathlib import Path
import sys
import click

from loki import (
    config as loki_config, Sourcefile, Frontend, as_tuple,
    set_excepthook, auto_post_mortem_debugger, info
)
from loki.batch import Scheduler, SchedulerConfig, ProcessingStrategy

from loki.transformations.build_system import FileWriteTransformation
from loki.transformations import SanitiseTransformation, SCCLowLevelHoist
from loki.transformations.transpile import FortranCTransformation, FortranISOCWrapperTransformation
from loki.transformations.build_system import DependencyTransformation

@click.group()
@click.option('--debug/--no-debug', default=False, show_default=True,
              help=('Enable / disable debug mode. This automatically attaches '
                    'a debugger when exceptions occur'))
def cli(debug):
    if debug:
        set_excepthook(hook=auto_post_mortem_debugger)


@cli.command()
@click.option('--mode', '-m', default='idem', type=click.STRING,
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
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--define', '-D', multiple=True,
              help='Additional symbol definitions for the C-preprocessor')
@click.option('--omni-include', type=click.Path(), multiple=True,
              help='Additional path for header files, specifically for OMNI')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional .xmod file(s) for OMNI')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--plan-file', type=click.Path(), default=None,
              help='Process pipeline in planning mode and generate CMake "plan" file.')
@click.option('--callgraph', '-g', type=click.Path(), default=None,
              help='Generate and display the subroutine callgraph.')
@click.option('--root', type=click.Path(), default=None,
              help='Root path to which all paths are relative to.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during batch processing')
def convert(
        mode, config, build, source, header, cpp, include, define,
        omni_include, xmod, frontend, plan_file, callgraph, root,
        log_level
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

    loki_config['log-level'] = log_level

    if plan_file is not None:
        processing_strategy = ProcessingStrategy.PLAN
        info(f'[Loki] Creating CMake plan file from config: {config}')
    else:
        processing_strategy = ProcessingStrategy.DEFAULT
        info(f'[Loki] Batch-processing source files using config: {config} ')

    print(f"processing_strategy: {processing_strategy}")
    print(f"plan_file: {plan_file}")

    config = SchedulerConfig.from_file(config)

    # set default transformation mode in Scheduler config
    config.default['mode'] = mode

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
    paths = [Path(p) for p in as_tuple(source)]
    paths += [Path(h).parent for h in as_tuple(header)]
    # Skip full source parse for planning mode
    full_parse = processing_strategy == ProcessingStrategy.DEFAULT
    scheduler = Scheduler(
        paths=paths, config=config, frontend=frontend,
        full_parse=full_parse, definitions=definitions,
        output_dir=build, **build_args
    )

    # If requested, apply a custom pipeline from the scheduler config
    # Note that this new entry point will bypass all other default
    # behaviour and exit immediately after.
    # if mode not in config.pipelines:
    #     msg = f'[Loki] ERROR: Pipeline or transformation mode {mode} not found in config file.\n'
    #     msg += '[Loki] Please provide a config file with configured transformation or pipelines instead.\n'
    #     sys.exit(msg)

    if mode in config.pipelines:
        info(f'[Loki-transform] Applying custom pipeline {mode} from config:')
        info(str(config.pipelines[mode]))

        scheduler.process(config.pipelines[mode], proc_strategy=processing_strategy)

        mode = mode.replace('-', '_')  # Sanitize mode string

        # Write out all modified source files into the build directory
        file_write_trafo = scheduler.config.transformations.get('FileWriteTransformation', None)
        if not file_write_trafo:
            file_write_trafo = FileWriteTransformation(cuf='cuf' in mode)
        scheduler.process(transformation=file_write_trafo, proc_strategy=processing_strategy)

        if plan_file is not None:
            scheduler.write_cmake_plan(plan_file, rootpath=root)

        if callgraph:
            scheduler.callgraph(callgraph)
        return

    # Pull dimension definition from configuration
    horizontal = scheduler.config.dimensions.get('horizontal', None)
    vertical = scheduler.config.dimensions.get('vertical', None)
    block_dim = scheduler.config.dimensions.get('block_dim', None)

    # First, remove all derived-type arguments; caller first!
    # if remove_derived_args and mode not in ['cuda-hoist']:
    #     scheduler.process( DerivedTypeArgumentsTransformation() )

    # Re-write DR_HOOK labels for non-GPU paths
    # if 'scc' not in mode and 'cuda' not in mode :
    #     scheduler.process( DrHookTransformation(mode=mode, remove=False) )

    # Perform general source removal of unwanted calls or code regions
    # (do not perfrom Dead Code Elimination yet, inlining will do this.)
    remove_code_trafo = scheduler.config.transformations.get('RemoveCodeTransformation', None)
    if not remove_code_trafo:
        remove_code_trafo = RemoveCodeTransformation(
            remove_marked_regions=True, remove_dead_code=False, kernel_only=True,
            call_names=('ABOR1', 'DR_HOOK'), intrinsic_names=('WRITE(NULOUT',)
        )
    scheduler.process(transformation=remove_code_trafo)

    # Perform general source sanitisation steps to level the playing field
    sanitise_trafo = scheduler.config.transformations.get('SanitiseTransformation', None)
    if not sanitise_trafo:
        sanitise_trafo = SanitiseTransformation(
            resolve_sequence_association=True # resolve_sequence_association,
        )
    scheduler.process(transformation=sanitise_trafo)

    # Perform source-inlining either from CLI arguments or from config
    # if False:
    #     inline_trafo = scheduler.config.transformations.get('InlineTransformation', None)
    #     if not inline_trafo:
    #         inline_trafo = InlineTransformation(
    #             inline_internals=inline_members, inline_marked=inline_marked,
    #             remove_dead_code=eliminate_dead_code, allowed_aliases=horizontal.index,
    #             resolve_sequence_association=resolve_sequence_association_inlined_calls 
    #         )
    #     scheduler.process(transformation=inline_trafo)

    # Backward insert argument shapes (for surface routines)
    # if derive_argument_array_shape and mode not in ['cuda-hoist']:
    #     scheduler.process(transformation=ArgumentArrayShapeAnalysis())
    #     scheduler.process(transformation=ExplicitArgumentArrayShapeTransformation())

    # Insert data offload regions for GPUs and remove OpenMP threading directives
    # if mode not in ['cuda-hoist', 'cuda-parametrise']:
    #     use_claw_offload = True
    #     if data_offload:
    #         offload_transform = DataOffloadTransformation(
    #             remove_openmp=remove_openmp, assume_deviceptr=assume_deviceptr
    #         )
    #         scheduler.process(offload_transform)
    #         use_claw_offload = not offload_transform.has_data_regions

    # if frontend == Frontend.OMNI and mode in ['idem-stack', 'scc-stack']:
    #     # To make the pool allocator size derivation work correctly, we need
    #     # to normalize the 1:end-style index ranges that OMNI introduces
    #     class NormalizeRangeIndexingTransformation(Transformation):
    #         def transform_subroutine(self, routine, **kwargs):
    #             normalize_range_indexing(routine)

    #     scheduler.process( NormalizeRangeIndexingTransformation() )

    # if global_var_offload and mode not in ['cuda-hoist']:
    #     scheduler.process(transformation=GlobalVariableAnalysis())
    #     scheduler.process(transformation=GlobalVarOffloadTransformation())

    # Now we create and apply the main transformation pipeline
    # if mode == 'idem':
    #     pipeline = IdemTransformation()
    #     scheduler.process( pipeline )

    # if mode == 'idem-stack':
    #     pipeline = Pipeline(
    #         classes=(IdemTransformation, TemporariesPoolAllocatorTransformation),
    #         block_dim=block_dim, directive='openmp', check_bounds=True
    #     )
    #     scheduler.process( pipeline )

    # if mode == 'idem-lower':
    #     pipeline = Pipeline(
    #         classes=(IdemTransformation,
    #             LowerBlockIndexTransformation,
    #             InjectBlockIndexTransformation,),
    #             # LowerBlockLoopTransformation),
    #         block_dim=block_dim, directive='openmp', check_bounds=True,
    #         horizontal=horizontal, vertical=vertical,
    #     )
    #     scheduler.process( pipeline )

    # if mode == 'idem-lower-loop':
    #     pipeline = Pipeline(
    #         classes=(IdemTransformation,
    #             LowerBlockIndexTransformation,
    #             InjectBlockIndexTransformation,
    #             LowerBlockLoopTransformation),
    #         block_dim=block_dim, directive='openmp', check_bounds=True,
    #         horizontal=horizontal, vertical=vertical,
    #     )
    #     scheduler.process( pipeline )

    # if mode == 'sca':
    #     pipeline = ExtractSCATransformation(horizontal=horizontal)
    #     scheduler.process( pipeline )

    # if mode == 'claw':
    #     pipeline = CLAWTransformation(
    #         horizontal=horizontal, claw_data_offload=use_claw_offload
    #     )
    #     scheduler.process( pipeline )

    # if mode == 'scc':
    #     pipeline = scheduler.config.transformations.get('scc', None)
    #     if not pipeline:
    #         pipeline = SCCVectorPipeline(
    #             horizontal=horizontal,
    #             block_dim=block_dim, directive=directive,
    #             trim_vector_sections=trim_vector_sections
    #         )
    #     scheduler.process( pipeline )

    # if mode == 'scc-hoist':
    #     pipeline = scheduler.config.transformations.get('scc-hoist', None)
    #     if not pipeline:
    #         pipeline = SCCHoistPipeline(
    #             horizontal=horizontal,
    #             block_dim=block_dim, directive=directive,
    #             dim_vars=(vertical.size,) if vertical else None,
    #             trim_vector_sections=trim_vector_sections
    #         )
    #     scheduler.process( pipeline )

    # if mode == 'scc-stack':
    #     pipeline = scheduler.config.transformations.get('scc-stack', None)
    #     if not pipeline:
    #         pipeline = SCCStackPipeline(
    #             horizontal=horizontal,
    #             block_dim=block_dim, directive=directive,
    #             check_bounds=False,
    #             trim_vector_sections=trim_vector_sections
    #         )
    #     scheduler.process( pipeline )

    # if mode == 'scc-raw-stack':
    #     pipeline = scheduler.config.transformations.get('scc-raw-stack', None)
    #     if not pipeline:
    #         pipeline = SCCStackPipeline(
    #             horizontal=horizontal,
    #             block_dim=block_dim, directive=directive,
    #             check_bounds=False,
    #             trim_vector_sections=trim_vector_sections,
    #         )
    #     scheduler.process( pipeline )

    # if mode in ['cuf-hoist']:
    #     pipeline = SCCLowLevelCufHoist(horizontal=horizontal, vertical=vertical, directive=directive, trim_vector_sections=trim_vector_sections,
    #             transformation_type='hoist', derived_types = ['TECLDP'], block_dim=block_dim,
    #             dim_vars=(vertical.size,), as_kwarguments=True, remove_vector_section=True)
    #     scheduler.process( pipeline )
    # 
    # if mode in ['cuf-parametrise']:
    #     dic2p = {'NLEV': 137}
    #     pipeline = SCCLowLevelCufParametrise(horizontal=horizontal, vertical=vertical, directive=directive, trim_vector_sections=trim_vector_sections,
    #             transformation_type='parametrise', derived_types = ['TECLDP'], block_dim=block_dim,
    #             dim_vars=(vertical.size,), as_kwarguments=True, dic2p=dic2p, remove_vector_section=True)
    #     scheduler.process( pipeline )

    if mode in ['cuda-hoist', 'hip-hoist']:
        dic2p = {'nang': 24, 'nfre': 36}
        tmp_mode = 'cuda'
        if 'hip' in mode:
            tmp_mode = 'hip'
        pipeline = SCCLowLevelHoist(horizontal=horizontal, vertical=vertical, directive='omp-gpu', trim_vector_sections=True, # trim_vector_sections,
                transformation_type='hoist', derived_types = ['TECLDP'], block_dim=block_dim, mode=tmp_mode, # 'cuda',
                # dim_vars=(vertical.size, horizontal.size),
                demote_local_arrays=True,
                as_kwarguments=True, hoist_parameters=True,
                ignore_modules=['parkind1'], all_derived_types=True,
                dic2p=dic2p, skip_driver_imports=True,
                allowed_aliases="IJ", inline_elementals=False,
                remove_marked_regions=True, remove_dead_code=False,
                call_names=['dr_hook', 'abort1'], intrinsic_names=['write(iu06'],
                kernel_only=True, dimensions=horizontal,
                loop_interchange=True)
        scheduler.process( pipeline )


    if mode in ['cuda-parametrise']:
        dic2p = {'NLEV': 137}
        pipeline = SCCLowLevelParametrise(horizontal=horizontal, vertical=vertical, directive=directive, trim_vector_sections=trim_vector_sections,
                transformation_type='parametrise', derived_types = ['TECLDP'], block_dim=block_dim, mode='cuda',
                # dim_vars=(vertical.size,),
                as_kwarguments=True, hoist_parameters=True, ignore_modules=['parkind1'], all_derived_types=True, dic2p=dic2p)
        scheduler.process( pipeline )

    mode = mode.replace('-', '_')  # Sanitize mode string
    if mode in ['c', 'cuda_parametrise', 'cuda_hoist', 'hip_hoist']:
        if mode in ['c']:
            f2c_transformation = FortranCTransformation(path=build)
        elif mode in ['cuda_parametrise', 'cuda_hoist', 'hip_hoist']:
            if 'cuda' in mode:
                f2c_transformation = FortranCTransformation(path=build, language='cuda') # , use_c_ptr=True)
                f2c_iso_wrapper_trafo = FortranISOCWrapperTransformation(language='cuda', use_c_ptr=True)
            else:
                f2c_transformation = FortranCTransformation(path=build, language='hip') # , use_c_ptr=True)
                f2c_iso_wrapper_trafo = FortranISOCWrapperTransformation(language='hip', use_c_ptr=True)
        else:
            assert False
        scheduler.process(f2c_transformation)
        scheduler.process(f2c_iso_wrapper_trafo)
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
        # builddir=build,
        # mode=mode,
        cuf='cuf' in mode,
        include_module_var_imports=True # global_var_offload
    ))


@cli.command('plan')
@click.option('--mode', '-m', default='idem', type=click.STRING,
              help='Transformation mode, selecting which code transformations to apply.')
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
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--callgraph', '-g', type=click.Path(), default=None,
              help='Generate and display the subroutine callgraph.')
@click.option('--plan-file', type=click.Path(),
              help='CMake "plan" file to generate.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during batch processing')
@click.pass_context
def plan(
        ctx, mode, config, header, source, build, root, cpp,
        frontend, callgraph, plan_file, log_level
):
    """
    Create a "plan", a schedule of files to inject and transform for a
    given configuration.
    """
    return ctx.forward(convert)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
