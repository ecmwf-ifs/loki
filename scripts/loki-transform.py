#!/usr/bin/env python

"""
Loki head script for source-to-source transformations concerning ECMWF
physics, including "Single Column" (SCA) and CLAW transformations.
"""

import sys
from pathlib import Path
import click

from loki import (
    SourceFile, Transformation, Transformer, FindNodes, Loop, Array, Pragma,
    DependencyTransformation, FortranCTransformation, Frontend, OMNI, OFP, JoinableStringList
)

# Bootstrap the locally defined transformations directory
sys.path.insert(0, str(Path(__file__).parent))
# pylint: disable=wrong-import-position,wrong-import-order
from transformations import DerivedTypeArgumentsTransformation
from transformations import SCATransformation


def get_typedefs(typedef, xmods=None, frontend=OFP):
    """
    Read derived type definitions from typedef modules.
    """
    definitions = {}
    for tfile in typedef:
        source = SourceFile.from_file(tfile, xmods=xmods, frontend=frontend)
        definitions.update(source.modules[0].typedefs)
    return definitions


class Dimension:
    """
    Dimension that defines a one-dimensional iteration space.

    :param name: Name of the dimension, as used in data array declarations
    :param variable: Name of the iteration variable used in loops in this
                     dimension.
    :param iteration: Tuple defining the start/end variable names or values for
                      loops in this dimension.
    """

    def __init__(self, name=None, aliases=None, variable=None, iteration=None):
        self.name = name
        self.aliases = aliases
        self.variable = variable
        self.iteration = iteration

    @property
    def variables(self):
        return (self.name, self.variable) + self.iteration

    @property
    def size_expressions(self):
        """
        Return a list of expression strings all signifying "dimension size".
        """
        iteration = ['%s - %s + 1' % (self.iteration[1], self.iteration[0])]
        # Add ``1:x`` size expression for OMNI (it will insert an explicit lower bound)
        iteration += ['1:%s - %s + 1' % (self.iteration[1], self.iteration[0])]
        iteration += ['1:%s' % self.name]
        iteration += ['1:%s' % alias for alias in self.aliases]
        return [self.name] + self.aliases + iteration

    @property
    def index_expressions(self):
        """
        Return a list of expression strings all signifying potential
        dimension indices, including range accesses like `START:END`.
        """
        i_range = ['%s:%s' % (self.iteration[0], self.iteration[1])]
        # A somewhat strange expression used in VMASS bracnhes
        i_range += ['%s-%s+1' % (self.variable, self.iteration[0])]
        return [self.variable] + i_range


def insert_claw_directives(routine, driver, claw_scalars, target):
    """
    Insert the necessary pragmas and directives to instruct the CLAW.

    Note: Must be run after generic SCA conversion.
    """
    # Insert loop pragmas in driver (in-place)
    for loop in FindNodes(Loop).visit(driver.body):
        if str(loop.variable).upper() == target.variable:
            pragma = Pragma(keyword='claw', content='sca forward create update')
            loop._update(pragma=pragma)

    # Generate CLAW directives and insert into routine spec
    segmented_scalars = JoinableStringList(claw_scalars, sep=', ', width=80, cont=' &\n & ')
    directives = [Pragma(keyword='claw', content='define dimension jl(1:nproma) &'),
                  Pragma(keyword='claw', content='sca &'),
                  Pragma(keyword='claw', content='scalar(%s)\n\n\n' % segmented_scalars)]
    routine.spec.append(directives)


def remove_omp_do(routine):
    """
    Utility routine that strips existing !$opm do pragmas from driver code.
    """
    mapper = {}
    for p in FindNodes(Pragma).visit(routine.body):
        if p.keyword.lower() == 'omp':
            if p.content.startswith('do') or p.content.startswith('end do'):
                mapper[p] = None
    routine.body = Transformer(mapper).visit(routine.body)


@click.group()
def cli():
    pass


@cli.command('idem')
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated source files.')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--flatten-args/--no-flatten-args', default=True,
              help='Flag to trigger derived-type argument unrolling')
@click.option('--openmp/--no-openmp', default=False,
              help='Flag to force OpenMP pragmas onto existing horizontal loops')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
def idempotence(out_path, source, driver, header, xmod, include, flatten_args, openmp, frontend):
    """
    Idempotence: A "do-nothing" debug mode that performs a parse-and-unparse cycle.
    """
    driver_name = 'CLOUDSC_DRIVER'
    kernel_name = 'CLOUDSC'

    frontend = Frontend[frontend.upper()]
    typedefs = get_typedefs(header, xmods=xmod,
                            frontend=OFP if frontend == OMNI else frontend)
    kernel = SourceFile.from_file(source, xmods=xmod, includes=include,
                                  frontend=frontend, typedefs=typedefs,
                                  builddir=out_path)
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=frontend, builddir=out_path)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    class IdemTransformation(Transformation):
        """
        Define a custom transformation pipeline that optionally inserts
        experimental OpenMP pragmas for horizontal loops.
        """

        def transform_subroutine(self, routine, **kwargs):
            # Define the horizontal dimension
            horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                                   variable='JL', iteration=('KIDIA', 'KFDIA'))

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
        # Caller must go first, as it needs info from routine
        driver.apply(DerivedTypeArgumentsTransformation(), role='driver')
        kernel.apply(DerivedTypeArgumentsTransformation(), role='kernel')

    # Now we instantiate our pipeline and apply the "idempotence" changes
    kernel.apply(IdemTransformation())
    driver.apply(IdemTransformation())

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_IDEM', mode='module', module_suffix='_MOD')
    kernel.apply(dependency, role='kernel')
    kernel.write(path=Path(out_path)/kernel.path.with_suffix('.idem.F90').name)

    # Re-generate the driver that mimicks the original source file,
    # but imports and calls our re-generated kernel.
    driver.apply(dependency, role='driver', targets=kernel_name)
    driver.write(path=Path(out_path)/driver.path.with_suffix('.idem.F90').name)


@cli.command()
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated souce files.')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--strip-omp-do', is_flag=True, default=False,
              help='Removes existing !$omp do loop pragmas')
@click.option('--mode', '-m', default='sca',
              type=click.Choice(['sca', 'claw']))
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
def convert(out_path, source, driver, header, xmod, include, strip_omp_do, mode, frontend):
    """
    Single Column Abstraction (SCA): Convert kernel into single-column
    format and adjust driver to apply it over in a horizontal loop.

    Optionally, this can also insert CLAW directives that may be use
    for further downstream transformations.
    """
    driver_name = 'CLOUDSC_DRIVER'
    kernel_name = 'CLOUDSC'

    frontend = Frontend[frontend.upper()]
    typedefs = get_typedefs(header, xmods=xmod,
                            frontend=OFP if frontend == OMNI else frontend)
    kernel = SourceFile.from_file(source, xmods=xmod, includes=include,
                                  frontend=frontend, typedefs=typedefs,
                                  builddir=out_path)
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=frontend, builddir=out_path)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    if mode == 'claw':
        claw_scalars = [v.name.lower() for v in kernel[kernel_name].variables
                        if isinstance(v, Array) and len(v.dimensions.index_tuple) == 1]

    # First, remove all derived-type arguments; caller first!
    driver.apply(DerivedTypeArgumentsTransformation(), role='driver')
    kernel.apply(DerivedTypeArgumentsTransformation(), role='kernel')

    # Define the target dimension to strip from kernel and caller
    horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                           variable='JL', iteration=('KIDIA', 'KFDIA'))

    # Now we instantiate our SCA pipeline and apply the changes
    sca_transform = SCATransformation(dimension=horizontal)
    driver.apply(sca_transform, role='driver')
    kernel.apply(sca_transform, role='kernel')

    if mode == 'claw':
        insert_claw_directives(kernel[kernel_name], driver[driver_name],
                               claw_scalars, target=horizontal)

    if strip_omp_do:
        remove_omp_do(driver[driver_name])

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_{}'.format(mode.upper()),
                                          mode='module', module_suffix='_MOD')
    kernel.apply(dependency, role='kernel')
    kernel.write(path=Path(out_path)/kernel.path.with_suffix('.%s.F90' % mode).name)

    # Re-generate the driver that mimicks the original source file,
    # but imports and calls our re-generated kernel.
    driver.apply(dependency, role='driver', targets=kernel_name)
    driver.write(path=Path(out_path)/driver.path.with_suffix('.%s.F90' % mode).name)


@cli.command()
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated souce files.')
@click.option('--header', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
def transpile(out_path, header, source, driver, xmod, include):
    """
    Convert kernels to C and generate ISO-C bindings and interfaces.
    """
    driver_name = 'CLOUDSC_DRIVER'
    kernel_name = 'CLOUDSC'

    # Parse original driver and kernel routine, and enrich the driver
    typedefs = get_typedefs(header, xmods=xmod, frontend=OFP)
    kernel = SourceFile.from_file(source, xmods=xmod, includes=include,
                                  frontend=OMNI, typedefs=typedefs,
                                  builddir=out_path)
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=OMNI, builddir=out_path)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    # First, remove all derived-type arguments; caller first!
    driver.apply(DerivedTypeArgumentsTransformation(), role='driver')
    kernel.apply(DerivedTypeArgumentsTransformation(), role='kernel')

    typepaths = [Path(h) for h in header]
    typemods = [SourceFile.from_file(tp, frontend=OFP)[tp.stem] for tp in typepaths]
    for typemod in typemods:
        FortranCTransformation().apply(source=typemod, path=out_path)

    # Now we instantiate our pipeline and apply the changes
    transformation = FortranCTransformation(header_modules=typemods)
    transformation.apply(kernel, path=out_path)

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
