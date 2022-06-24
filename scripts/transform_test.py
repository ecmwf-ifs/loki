#!/usr/bin/env python

"""
Loki head script for source-to-source transformations concerning ECMWF
physics, including "Single Column" (SCA) and CLAW transformations.
"""

import sys
from pathlib import Path
import click

from loki import (
    Sourcefile, Transformation, Scheduler, SchedulerConfig,
    Frontend, as_tuple, auto_post_mortem_debugger, flatten, info
)

# Get generalized transformations provided by Loki
from loki.transform import (
    DependencyTransformation, FortranCTransformation, CMakePlanner
)

# Bootstrap the local transformations directory for custom transformations
sys.path.insert(0, str(Path(__file__).parent))

from transformations import DerivedTypeArgumentsTransformation, InferArgShapeTransformation
from transformations import DataOffloadTransformation
from transformations import ExtractSCATransformation, CLAWTransformation
from transformations import SingleColumnCoalescedTransformation
from transformations import DrHookTransformation

test_config = {
    'default': {
        'mode': 'idem',
        'role': 'kernel',
        'expand': True,
        'strict': True,
        # Ensure that we are never adding these to the tree, and thus
        # do not attempt to look up the source files for these.
        # TODO: Add type-bound procedure support and adjust scheduler to it
        'disable': ['timer%start', 'timer%end', 'timer%thread_start', 'timer%thread_end',
                    'timer%thread_log', 'timer%thread_log', 'timer%print_performance']
    },
    'routine': [
        {
            'name': 'test_driver',
            'role': 'driver',
            'expand':True,
        }
    ],
    'dimension': [
        {
            'name': 'horizontal',
            'size': 'KLON',
            'index': 'JL',
            'bounds': ('KIDIA', 'KFDIA'),
            'aliases': ['NPROMA', 'KDIM%KLON'],
        },
        {
            'name': 'vertical',
            'size': 'KLEV',
            'index': 'JK',
        },
        {
            'name': 'block_dim',
            'size': 'NGPBLKS',
            'index': 'IBL',
        }
    ]
}


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
        sys.excepthook = auto_post_mortem_debugger

@cli.command()
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated souce files.')
@click.option('--path', '-p', type=click.Path(),
              help='Path to search during source exploration.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--data-offload', is_flag=True, default=False,
              help='Run transformation to insert custom data offload regions.')
@click.option('--mode', '-m', default='idem',
              type=click.Choice(['idem', 'sca', 'claw', 'scc', 'scc-hoist']),
              help='Transformation mode, selecting which code transformations to apply.')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--config', default=None, type=click.Path(),
              help='Path to custom scheduler configuration file')
def convert(out_path, path,header,data_offload,mode, frontend, config):
    """
    Single Column Abstraction (SCA): Convert kernel into single-column
    format and adjust driver to apply it over in a horizontal loop.

    Optionally, this can also insert CLAW directives that may be use
    for further downstream transformations.
    """
    if config is None:
        config = SchedulerConfig.from_dict(test_config)
    else:
        config = SchedulerConfig.from_file(config)

    build_args = {
#        'preprocess': cpp,
#        'includes': include,
#        'defines': define,
#        'xmods': xmod,
#        'omni_includes': omni_include,
    }

    frontend = Frontend[frontend.upper()]
    frontend_type = Frontend.FP if frontend == Frontend.OMNI else frontend

    # Note, in order to get function inlinig correct, we need full knowledge
    # of any imported symbols and functions. Since we cannot yet retro-fit that
    # after creation, we need to make sure that the order of definitions can
    # be used to create a coherent stack of type definitions.

    definitions = []
    for h in header:
        sfile = Sourcefile.from_file(h, frontend=frontend_type, **build_args)
        definitions = definitions + list(sfile.modules)

    # Create a scheduler to bulk-apply source transformations
    paths = [Path(p).resolve() for p in as_tuple(path)]
    scheduler = Scheduler(paths=paths, config=config, frontend=frontend,
                          definitions=definitions, **build_args)
    scheduler.populate(routines=config.routines.keys())

    # First, remove all derived-type arguments; caller first!
    scheduler.process(transformation=DerivedTypeArgumentsTransformation())

    # Insert data offload regions for GPUs and remove OpenMP threading directives
    use_claw_offload = True
    if data_offload:
        offload_transform = DataOffloadTransformation(remove_openmp=remove_openmp)
        scheduler.process(transformation=offload_transform)
        use_claw_offload = not offload_transform.has_data_regions

    # Now we instantiate our transformation pipeline and apply the main changes
    transformation = None
    if mode == 'idem':
        transformation = IdemTransformation()

    if mode == 'sca':
        horizontal = scheduler.config.dimensions['horizontal']
        transformation = ExtractSCATransformation(horizontal=horizontal)

    if mode in ['scc', 'scc-hoist']:
        horizontal = scheduler.config.dimensions['horizontal']
        vertical = scheduler.config.dimensions['vertical']
        block_dim = scheduler.config.dimensions['block_dim']
        transformation = SingleColumnCoalescedTransformation(
            horizontal=horizontal, vertical=vertical, block_dim=block_dim,
            directive='openacc', hoist_column_arrays='hoist' in mode
        )
 
    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    mode = mode.replace('-', '_')  # Sanitize mode string
    dependency = DependencyTransformation(suffix=f'_{mode.upper()}',
                                          mode='module', module_suffix='_MOD')

    scheduler.process(transformation=dependency)
    if transformation:
        scheduler.process(transformation=transformation)
    else:
        raise RuntimeError('[Loki] Convert could not find specified Transformation!')


    # Write out all modified source files into the build directory
    for item in scheduler.items:
        suffix = f'.{mode}.F90'
        sourcefile = item.source
        sourcefile.write(path=Path(out_path)/sourcefile.path.with_suffix(suffix).name)

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
