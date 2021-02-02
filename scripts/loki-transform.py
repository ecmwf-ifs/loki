#!/usr/bin/env python

"""
Loki head script for source-to-source transformations concerning ECMWF
physics, including "Single Column" (SCA) and CLAW transformations.
"""

import sys
from pathlib import Path
import click

from loki import (
    Sourcefile, Module, Transformation, Transformer, Scheduler,
    FindNodes, Loop, Pragma, Frontend, flatten, Dimension
)

# Get generalized transformations provided by Loki
from loki.transform import DependencyTransformation, FortranCTransformation

# Bootstrap the local transformations directory for custom transformations
sys.path.insert(0, str(Path(__file__).parent))
# pylint: disable=wrong-import-position,wrong-import-order
from transformations import DerivedTypeArgumentsTransformation
from transformations import ExtractSCATransformation, CLAWTransformation
from transformations import DataOffloadTransformation


"""
Scheduler configuration for the CLOUDSC ESCAPE dwarf.

This defines the "roles" of the two main source files
("driver" and "kernel") and adds exemptions for the
bulk-processing scheduler to ignore the timing utlitiies.
"""
cloudsc_config = {
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
            'name': 'cloudsc_driver',
            'role': 'driver',
            'expand': True,
        }
    ]
}


"""
Define the horizontal dimension used in the CLOUDSC demonstrator
dwarf in terms of expression strings. These expressions are then
used to extract SCA-format code for Loki-CLAW transformations.
"""
horizontal = Dimension(name='horizontal', size='KLON', index='JL',
                       bounds=('KIDIA', 'KFDIA'),
                       aliases=['NPROMA', 'KDIM%KLON'])


@click.group()
def cli():
    pass


@cli.command('idem')
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated source files.')
@click.option('--source', '-s', type=click.Path(), multiple=True,
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--define', '-I', multiple=True,
              help='Additional symbol definitions for C-preprocessor')
@click.option('--omni-include', '-I', type=click.Path(), multiple=True,
              help='Additional path for header files, specifically for OMNI')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--flatten-args/--no-flatten-args', default=True,
              help='Flag to trigger derived-type argument unrolling')
@click.option('--openmp/--no-openmp', default=False,
              help='Flag to force OpenMP pragmas onto existing horizontal loops')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--config', default=None, type=click.Path(),
              help='Path to custom scheduler configuration file')
def idempotence(out_path, source, driver, header, cpp, include, define, omni_include, xmod,
                flatten_args, openmp, frontend, config):
    """
    Idempotence: A "do-nothing" debug mode that performs a parse-and-unparse cycle.
    """
    if config is None:
        config = cloudsc_config

    frontend = Frontend[frontend.upper()]
    frontend_type = Frontend.OFP if frontend == Frontend.OMNI else frontend
    definitions = flatten(Sourcefile.from_file(h, xmods=xmod,
                                               frontend=frontend_type).modules for h in header)

    # Create a scheduler to bulk-apply source transformations
    paths = [Path(s).resolve().parent for s in source]
    paths += [Path(h).resolve().parent for h in header]
    scheduler = Scheduler(paths=paths, config=config, defines=define, definitions=definitions)
    scheduler.populate(routines=[r['name'] for r in config['routine']])

    class IdemTransformation(Transformation):
        """
        Define a custom transformation pipeline that optionally inserts
        experimental OpenMP pragmas for horizontal loops.
        """

        def transform_subroutine(self, routine, **kwargs):
            if openmp:
                # Experimental OpenMP loop pragma insertion
                for loop in FindNodes(Loop).visit(routine.body):
                    if loop.variable == horizontal.variable:
                        # Update the loop in-place with new OpenMP pragmas
                        pragma = Pragma(keyword='omp', content='do simd')
                        pragma_nowait = Pragma(keyword='omp',
                                               content='end do simd nowait')
                        loop._update(pragma=pragma, pragma_post=pragma_nowait)

    if flatten_args:
        # Unroll derived-type arguments into multiple arguments
        scheduler.process(transformation=DerivedTypeArgumentsTransformation())

    # Now we instantiate our pipeline and apply the "idempotence" changes
    scheduler.process(transformation=IdemTransformation())

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_IDEM', mode='module', module_suffix='_MOD')
    scheduler.process(transformation=dependency)

    # Write out all modified source files into the build directory
    for item in scheduler.items:
        sourcefile = item.source
        sourcefile.write(path=Path(out_path)/sourcefile.path.with_suffix('.idem.F90').name)


@cli.command()
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated souce files.')
@click.option('--source', '-s', type=click.Path(), multiple=True,
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--define', '-I', multiple=True,
              help='Additional symbol definitions for C-preprocessor')
@click.option('--omni-include', '-I', type=click.Path(), multiple=True,
              help='Additional path for header files, specifically for OMNI')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--remove-openmp', is_flag=True, default=False,
              help='Removes existing OpenMP pragmas in "!$loki data" regions')
@click.option('--mode', '-m', default='sca',
              type=click.Choice(['sca', 'claw']))
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--config', default=None, type=click.Path(),
              help='Path to custom scheduler configuration file')
def convert(out_path, source, driver, header, cpp, include, define, omni_include, xmod,
            remove_openmp, mode, frontend, config):
    """
    Single Column Abstraction (SCA): Convert kernel into single-column
    format and adjust driver to apply it over in a horizontal loop.

    Optionally, this can also insert CLAW directives that may be use
    for further downstream transformations.
    """
    if config is None:
        config = cloudsc_config

    frontend = Frontend[frontend.upper()]
    frontend_type = Frontend.OFP if frontend == Frontend.OMNI else frontend
    definitions = flatten(Sourcefile.from_file(h, xmods=xmod,
                                               frontend=frontend_type).modules for h in header)

    # Create a scheduler to bulk-apply source transformations
    paths = [Path(s).resolve().parent for s in source]
    paths += [Path(h).resolve().parent for h in header]
    scheduler = Scheduler(paths=paths, config=config, defines=define, definitions=definitions)
    scheduler.populate(routines=[r['name'] for r in config['routine']])

    # First, remove all derived-type arguments; caller first!
    scheduler.process(transformation=DerivedTypeArgumentsTransformation())

    use_claw_offload = False
    if mode == 'claw':
        offload_transform = DataOffloadTransformation(remove_openmp=remove_openmp)
        scheduler.process(transformation=offload_transform)
        use_claw_offload = not offload_transform.has_data_regions

    # Now we instantiate our SCA pipeline and apply the changes
    if mode == 'sca':
        sca_transform = ExtractSCATransformation(horizontal=horizontal)
    elif mode == 'claw':
        sca_transform = CLAWTransformation(
            horizontal=horizontal, claw_data_offload=use_claw_offload
        )
    scheduler.process(transformation=sca_transform)

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_{}'.format(mode.upper()),
                                          mode='module', module_suffix='_MOD')
    scheduler.process(transformation=dependency)

    # Write out all modified source files into the build directory
    for item in scheduler.items:
        suffix = '.{}.F90'.format(mode)
        sourcefile = item.source
        sourcefile.write(path=Path(out_path)/sourcefile.path.with_suffix(suffix).name)


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
    frontend_type = Frontend.OFP if frontend == Frontend.OMNI else frontend

    # Note, in order to get function inlinig correct, we need full knowledge
    # of any imported symbols and functions. Since we cannot yet retro-fit that
    # after creation, we need to make sure that the order of definitions can
    # be used to create a coherent stack of type definitions.
    definitions = []
    for h in header:
        sfile = Sourcefile.from_file(h, xmods=xmod, definitions=definitions,
                                     frontend=frontend_type)
        definitions = definitions + list(sfile.modules)

    # Parse original driver and kernel routine, and enrich the driver
    kernel = Sourcefile.from_file(source, definitions=definitions, preprocess=cpp,
                                  includes=include, defines=define, xmods=xmod,
                                  frontend=frontend)
    driver = Sourcefile.from_file(driver, xmods=xmod, frontend=frontend)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    # First, remove all derived-type arguments; caller first!
    driver.apply(DerivedTypeArgumentsTransformation(), role='driver')
    kernel.apply(DerivedTypeArgumentsTransformation(), role='kernel')

    # Now we instantiate our pipeline and apply the changes
    transformation = FortranCTransformation()
    transformation.apply(kernel, role='kernel', path=out_path)

    # Traverse header modules to create getter functions for module variables
    for header in definitions:
        transformation.apply(header, role='header', path=out_path)

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_FC', mode='module', module_suffix='_MOD')
    kernel.apply(dependency, role='kernel')
    kernel.write(path=Path(out_path)/kernel.path.with_suffix('.c.F90').name)

    # Re-generate the driver that mimicks the original source file,
    # but imports and calls our re-generated kernel.
    driver.apply(dependency, role='driver', targets=kernel_name)
    driver.write(path=Path(out_path)/driver.path.with_suffix('.c.F90').name)


if __name__ == "__main__":
    cli()
