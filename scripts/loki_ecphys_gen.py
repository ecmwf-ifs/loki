#!/usr/bin/env python

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
An offline script to generate the complex control-flow patterns of the
EC-physics drivers.
"""

from pathlib import Path
import click
from codetiming import Timer

from loki import config as loki_config, Sourcefile, Dimension, info
from loki.analyse import dataflow_analysis_attached
from loki.expression import symbols as sym, parse_expr
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, SubstituteExpressions,
    SubstituteStringExpressions, Transformer, CallStatement,
    pragmas_attached, pragma_regions_attached, is_loki_pragma,
    get_pragma_parameters
)
from loki.logging import warning
from loki.scope import SymbolAttributes
from loki.tools import as_tuple, flatten
from loki.types import DerivedType, BasicType

from loki.transformations.inline import inline_marked_subroutines
from loki.transformations.sanitise import (
    merge_associates, resolve_associates,
    transform_sequence_association_append_map
)
from loki.transformations.remove_code import (
    do_remove_marked_regions, do_remove_unused_imports
)
from loki.transformations.drhook import DrHookTransformation
from loki.transformations.extract import outline_region
from loki.transformations.build_system import ModuleWrapTransformation
from loki.transformations.parallel import (
    remove_openmp_regions, add_openmp_pragmas,
    remove_explicit_firstprivatisation,
    create_explicit_firstprivatisation
)
from loki.transformations.sanitise import ResolveAssociatesTransformer


# List of types that we know to be FIELD API groups
field_group_types = [
    'FIELD_VARIABLES', 'STATE_TYPE', 'MODEL_STATE_TYPE',
    'PERTURB_TYPE', 'AUX_TYPE', 'AUX_RAD_TYPE', 'FLUX_TYPE',
    'AUX_DIAG_TYPE', 'AUX_DIAG_LOCAL_TYPE', 'DDH_SURF_TYPE',
    'SURF_AND_MORE_LOCAL_TYPE', 'KEYS_LOCAL_TYPE',
    'PERTURB_LOCAL_TYPE', 'GEMS_LOCAL_TYPE',
    'FIELD_3RB_ARRAY', 'FIELD_4RB_ARRAY', 'ECPHYS_OPTS_TYPE'
]

fgroup_dimension = ['DIMENSION_TYPE']
fgroup_firstprivates = ['SURF_AND_MORE_TYPE']
lcopies_firstprivates = {'ZSURF': 'ZSURFACE'}

# List of variables that we know to have global scope
global_variables = [
    'PGFL', 'PGFLT1', 'YDGSGEOM', 'YDMODEL',
    'YDDIM', 'YDSTOPH', 'YDGEOMETRY',
    'YDSURF', 'YDGMV', 'SAVTEND',
    'YGFL', 'PGMV', 'PGMVT1', 'ZGFL_DYN',
    'ZCONVCTY', 'YDDIMV', 'YDPHY2',
    'PHYS_MWAVE', 'ZSPPTGFIX', 'ZSURFACE'
]


def remove_block_loops(routine):
    """
    Remove any outer block :any:`Loop` from a given :any:`Subroutine.
    """

    class RemoveBlockLoopTransformer(Transformer):
        """
        :any:`Transformer` to remove driver-level block loops.
        """

        def visit_Loop(self, loop, **kwargs):
            if not loop.variable == 'JKGLO':
                return loop

            to_remove = tuple(
                a for a in FindNodes(ir.Assignment).visit(loop.body)
                if a.lhs in ['ICST', 'ICEND', 'IBL']
            )
            return tuple(n for n in loop.body if n not in to_remove)

    routine.body = RemoveBlockLoopTransformer().visit(routine.body)


def remove_field_api_view_updates(routine, field_group_types):
    """
    Remove FIELD API boilerplate calls for view updates
    """

    class RemoveFieldAPITransformer(Transformer):

        def visit_CallStatement(self, call, **kwargs):

            if 'UPDATE_VIEW' in str(call.name):
                if not call.name.parent:
                    warning(f'[Loki::ControlFlow] Removing {call.name} call without parent!')
                if not str(call.name.parent.type.dtype) in field_group_types:
                    warning(f'[Loki::ControlFlow] Removing {call.name} call, but not in field group types!')

                return None

            if 'IDIMS%UPDATE' == str(call.name):
                return None

            return call

        def visit_Assignment(self, assign, **kwargs):
            if assign.lhs.type.dtype in field_group_types:
                warning(f'[Loki::ControlFlow] Found LHS field group assign: {assign}')
            return assign

        def visit_Loop(self, loop, **kwargs):
            loop = self.visit_Node(loop, **kwargs)
            return loop if loop.body else None

        def visit_Conditional(self, cond, **kwargs):
            cond = super().visit_Node(cond, **kwargs)
            return cond if cond.body else None

    routine.body = RemoveFieldAPITransformer().visit(routine.body)


def add_field_api_view_updates(routine, dimension, field_group_types):
    """
    Add FIELD API boilerplate calls for view updates
    """

    def _create_dim_update(scope):
        index = parse_expr(dimension.index, scope)
        upper = parse_expr(dimension.bounds_expressions[1][1], scope)
        bindex = parse_expr(dimension.index_expressions[1], scope)
        idims = scope.get_symbol('IDIMS')
        csym = sym.ProcedureSymbol(name='UPDATE', parent=idims, scope=idims.scope)
        return ir.CallStatement(name=csym, arguments=(bindex, upper, index), kwarguments=())

    def _create_view_updates(section, scope):
        bindex = parse_expr(dimension.index_expressions[1], scope)

        fgroup_vars = sorted(tuple(
            v for v in FindVariables(unique=True).visit(section)
            if str(v.type.dtype) in field_group_types
        ), key=lambda v: str(v))
        calls = ()
        for fgvar in fgroup_vars:
            fgsym = scope.get_symbol(fgvar.name)
            csym = sym.ProcedureSymbol(name='UPDATE_VIEW', parent=fgsym, scope=fgsym.scope)
            calls += (ir.CallStatement(name=csym, arguments=(bindex,), kwarguments=()),)

        return calls

    class InsertFieldAPIViewsTransformer(Transformer):
        """ Injects FIELD-API view updates into block loops """

        def visit_Loop(self, loop, **kwargs):
            if not loop.variable == 'JKGLO':
                return loop

            scope = kwargs.get('scope')

            # Find the loop-setup assignments
            _loop_symbols = ('JKGLO', 'IBL', 'ICST', 'ICEND')
            loop_setup = tuple(
                a for a in FindNodes(ir.Assignment).visit(loop.body)
                if a.lhs in _loop_symbols
            )
            idx = max(loop.body.index(a) for a in loop_setup) + 1

            # Prepend FIELD API boilerplate
            preamble = (
                ir.Comment(''), ir.Comment('! Set up thread-local view pointers')
            )
            preamble += (_create_dim_update(scope),)
            preamble += _create_view_updates(loop.body, scope)

            loop._update(body=loop.body[:idx] + preamble + loop.body[idx:])
            return loop

    routine.body = InsertFieldAPIViewsTransformer().visit(routine.body, scope=routine)


def add_block_loops(routine, dimension):
    """
    Insert IFS-style driver block-loops (NPROMA).
    """

    # TODO: The abuse of `Dimension` here includes some back-bending
    # hackery due to the funcky way in which the block-loop bounds are
    # done in IFS!

    # Ensure that local integer variables are declared
    index = parse_expr(dimension.index, routine)
    upper = parse_expr(dimension.bounds_expressions[1][1], routine)
    bidx = parse_expr(dimension.index_expressions[1], routine)
    for v in (index, upper, bidx):
        if not v in routine.variable_map:
            routine.variables += (
                v.clone(type=SymbolAttributes(BasicType.INTEGER, kind='JPIM')),
            )

    def _create_block_loop(body, scope):
        """
        Generate block loop object, including indexing preamble
        """

        # This is a hack; it's meant to be the upper limit, but we use it as stride!
        bsize = parse_expr(dimension.bounds_expressions[1][0], scope=scope)
        size = parse_expr(dimension.size, scope=scope)
        lrange = sym.LoopRange((sym.Literal(1), size, bsize))

        expr_tail = parse_expr(f'{size}-{index}+1', scope=scope)
        expr_max = sym.InlineCall(
            function=sym.ProcedureSymbol('MIN', scope=scope), parameters=(bsize, expr_tail)
        )
        preamble = (ir.Assignment(lhs=upper, rhs=expr_max),)
        preamble += (ir.Assignment(
            lhs=bidx, rhs=parse_expr(f'({index}-1)/{bsize}+1', scope=scope)
        ),)

        return ir.Loop(variable=index, bounds=lrange, body=preamble + body)

    class InsertBlockLoopTransformer(Transformer):
        """ Creates a block loop per marked parallel region """

        def visit_PragmaRegion(self, region, **kwargs):
            """
            (Re-)insert driver-level block loops into marked parallel region.
            """
            if not is_loki_pragma(region.pragma, starts_with='parallel'):
                return region

            scope = kwargs.get('scope')

            loop = _create_block_loop(body=region.body, scope=scope)

            region._update(body=(ir.Comment(''), loop))
            return region

    with pragma_regions_attached(routine):
        routine.body = InsertBlockLoopTransformer().visit(routine.body, scope=routine)


def remove_redundant_declarations(routine):
    """
    Removes all local symbol declarations that are not being used in
    the routine body.
    """
    used_symbols = FindVariables(unique=True).visit(routine.body)
    # used_symbols |= {v.parents for v in used_symbols}
    used_symbols = tuple(v.name for v in used_symbols)

    decl_map = {}
    for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
        # Filter out routine arguments; we don't want to change the signature
        if any(s.name.lower() in routine._dummies for s in decl.symbols):
            continue

        # Filter out variables that are not used in the routine body
        symbols = tuple(s for s in decl.symbols if s.name in used_symbols)
        if symbols == decl.symbols:
            continue

        # Remove if no symbols are used, otherwise strip unused ones
        decl_map[decl] = decl.clone(symbols=symbols) if symbols else None
    routine.spec = Transformer(decl_map).visit(routine.spec)


def promote_temporary_arrays(routine, horizontal, blocking):
    """
    Promote remaining block-scoped local temporary arrays to full size.
    """
    block_size = routine.resolve_typebound_var(blocking.size)
    block_idx = routine.resolve_typebound_var(blocking.index)

    arrays_to_promote = tuple(
        v.name for v in routine.variables
        if isinstance(v, sym.Array) and \
        not v.name in routine._dummies and \
        v.shape[0] == horizontal.size and \
        not v.shape[-1] == blocking.size
    )

    # First, update the body symbols (which requires the shape)
    vmap = {}
    for var in FindVariables(unique=False).visit(routine.body):
        if var.name not in arrays_to_promote:
            continue
        if var.shape and block_size in var.shape:
            continue

        if var.dimensions:
            new_dims = var.dimensions + (block_idx,)
        else:
            new_dims = tuple(sym.Range((None, None)) for _ in var.shape) + (block_idx,)
        vmap[var] = var.clone(dimensions=new_dims)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Then update the declaration and the shape with it
    for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
        if not any(s.name in arrays_to_promote for s in decl.symbols):
            continue

        symbols = tuple(
            s.clone(
                dimensions=s.dimensions+(block_size,),
                type=s.type.clone(shape=s.type.shape+(block_size,))
            )
            if s.name in arrays_to_promote else s
            for s in decl.symbols
        )
        decl._update(symbols=symbols)


def extract_driver_routines(routine):
    """
    Extracts driver routines and replaces them with an appropriate
    :any:`CallStatement`.
    """
    imports = FindNodes(ir.Import).visit(routine.spec)
    mapper = {}
    driver_routines = []
    parent_vmap = routine.variable_map
    with pragma_regions_attached(routine):
        for region in FindNodes(ir.PragmaRegion).visit(routine.body):
            if not is_loki_pragma(region.pragma, starts_with='extract'):
                continue

            # Resolve associations in the local region before processing
            ResolveAssociatesTransformer(inplace=True).visit(region)

        with dataflow_analysis_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if not is_loki_pragma(region.pragma, starts_with='extract'):
                    continue

                # Name the external routine
                parameters = get_pragma_parameters(region.pragma, starts_with='extract')
                name = parameters['name']

                intent_map = {}
                intent_map['in'] = tuple(parent_vmap[v.lower()] for v in parameters.get('in', '').split(',') if v)
                intent_map['inout'] = tuple(parent_vmap[v.lower()] for v in parameters.get('inout', '').split(',') if v)
                intent_map['out'] = tuple(parent_vmap[v.lower()] for v in parameters.get('out', '').split(',') if v)

                call, region_routine = outline_region(region, name, imports, intent_map=intent_map)

                do_remove_unused_imports(region_routine)

                driver_routines.append(region_routine)

                # Replace region by call in original routine
                mapper[region] = call

            routine.body = Transformer(mapper=mapper).visit(routine.body)

    return driver_routines

@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), default=Path.cwd(),
              help='Path to search for initial input sources.')
@click.option('--build', '-b', '--out', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--remove-openmp/--no-remove-openmp', default=True,
              help='Flag to replace OpenMP loop annotations with Loki pragmas.')
@click.option('--sanitize-assoc/--no-sanitize-assoc', default=True,
              help='Flag to trigger ASSOCIATE block sanitisation.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during processing')
def inline(source, build, remove_openmp, sanitize_assoc, log_level):
    """
    Inlines EC_PHYS and CALLPAR into EC_PHYS_DRV to expose the parallel loop.
    """
    loki_config['log-level'] = log_level

    source = Path(source)
    build = Path(build)

    # Get everything set up...
    ec_phys_drv = Sourcefile.from_file(source/'ec_phys_drv.F90')['EC_PHYS_DRV']
    ec_phys = Sourcefile.from_file(source/'ec_phys.F90')['EC_PHYS']
    callpar = Sourcefile.from_file(source/'callpar.F90')['CALLPAR']

    ec_phys_drv.enrich(ec_phys)
    ec_phys.enrich(callpar)

    # Clone original and change subroutine name
    ec_phys_fc = ec_phys_drv.clone(name='EC_PHYS_FC')

    # Substitute symbols that do not exist in the caller context after inlining
    ec_phys.spec = SubstituteStringExpressions({
        'DIMS%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'DIMS%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
        'DIMS%KLEVS': 'YDSURF%YSP_SBD%NLEVS',
    }, scope=ec_phys).visit(ec_phys.spec)
    callpar.spec = SubstituteStringExpressions({
        'KDIM%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'KDIM%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
    }, scope=callpar).visit(callpar.spec)

    # Before inlining, remove DR_HOOK calls from the inner routines
    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Removed inner DR_HOOK calls in {s:.2f}s'):
        DrHookTransformation(kernel_only=False, remove=True).apply(ec_phys, role='driver')
        DrHookTransformation(kernel_only=False, remove=True).apply(callpar, role='driver')

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Inlined EC_PHYS in {s:.2f}s'):
        # Strip the outer block loop
        ecphys_calls = [
            c for c in FindNodes(ir.CallStatement).visit(ec_phys_fc.body) if c.name == 'EC_PHYS'
        ]

        # Ouch, this is horrible!
        call_map = {}
        transform_sequence_association_append_map(call_map, ecphys_calls[0])
        ec_phys_fc.body = Transformer(call_map).visit(ec_phys_fc.body)

        inline_marked_subroutines(ec_phys_fc)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Inlined CALLPAR in {s:.2f}s'):
        # Now just inline CALLPAR
        inline_marked_subroutines(ec_phys_fc, allowed_aliases=('JL', 'JK', 'J2D'))

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Remove marked regions in {s:.2f}s'):
        do_remove_marked_regions(ec_phys_fc)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Remove OpenMP regions in {s:.2f}s'):
        # Now remove OpenMP regions, as their symbols are not remapped
        remove_openmp_regions(ec_phys_fc)

    if not remove_openmp:
        # Re-insert OpenMP parallel regions after inlining
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-generated OpenMP regions in {s:.2f}s'):
            fgtypes = field_group_types + fgroup_dimension + fgroup_firstprivates
            add_openmp_pragmas(
                routine=ec_phys_fc,
                field_group_types=fgtypes,
                global_variables=global_variables
            )

    if sanitize_assoc:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Merged associate blocks in {s:.2f}s'):
            # First move all associatesion up to the outermost
            merge_associates(ec_phys_fc, max_parents=2)

        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Resolved associate blocks in {s:.2f}s'):
            # Then resolve all remaining inner associations
            resolve_associates(ec_phys_fc, start_depth=1)

    # Replace the docstring to mark routine as auto-generated
    ec_phys_fc.docstring = """
    !**** *EC_PHYS_FC* - Standaline physics forecast-only driver

    !     Purpose.
    !     --------
    !           An automatically generated driver routine that exposes
    !           parallel regions alongside the physics control flow for
    !           using different parallelisation methods, including GPU offload.

    !     **  THIS SUBROUTINE HAS BEEN AUTO-GENERATED BY LOKI  **

    !     It is a combination of EC_PHYS_DRV, EC_PHYS and CALLPAR and be re-derived from them.

"""

    # Rename DR_HOOK calls to ensure appropriate performance logging
    DrHookTransformation(
        rename={'EC_PHYS_DRV': 'EC_PHYS_FC'}, kernel_only=False
    ).apply(ec_phys_fc, role='driver')

    # Create source file, wrap as a module adjust DR_HOOK labels and write to file
    srcfile = Sourcefile(path=build/'ec_phys_fc_mod.F90', ir=(ec_phys_fc,))
    ModuleWrapTransformation(module_suffix='_MOD').apply(srcfile, role='kernel')

    srcfile.write()


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), default=Path.cwd(),
              help='Path to search for initial input sources.')
@click.option('--build', '-b', '--out', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--remove-block-loop/--no-remove-block-loop', default=True,
              help='Flag to replace OpenMP loop annotations with Loki pragmas.')
@click.option('--promote-local-arrays/--no-promote-local-arrays', default=True,
              help='Flag to promote local block-scope arrays to full size')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during processing')
def parallel(source, build, remove_block_loop, promote_local_arrays, log_level):
    """
    Generate parallel regions with OpenMP and OpenACC dispatch.
    """
    loki_config['log-level'] = log_level

    source = Path(source)
    build = Path(build)

    # Get everything set up...
    ec_phys_fc = Sourcefile.from_file(source/'ec_phys_fc_mod.F90')['EC_PHYS_FC']

    # Clone original and change subroutine name
    ec_phys_parallel = ec_phys_fc.clone(name='EC_PHYS_PARALLEL')

    blocking = Dimension(
        name='block', index='JKGLO', index_aliases='IBL',
        size='YDGEM%NGPTOT', aliases='YDDIM%NGPBLKS',
        bounds=('YDGEM%NGPTOT', 'YDDIM%NPROMA'),
        bounds_aliases=('ICST', 'ICEND')
    )

    if remove_block_loop:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-generated block loops in {s:.2f}s'):
            # Remove explicit firstprivatisation
            remove_explicit_firstprivatisation(
                ec_phys_parallel.body, fprivate_map=lcopies_firstprivates, routine=ec_phys_parallel
            )

            # Strip the outer block loop and FIELD-API boilerplate
            remove_block_loops(ec_phys_parallel)

            remove_field_api_view_updates(
                ec_phys_parallel, field_group_types=field_group_types+fgroup_firstprivates
            )

            # The add them back in according to parallel region
            add_block_loops(ec_phys_parallel, dimension=blocking)

            add_field_api_view_updates(
                ec_phys_parallel, dimension=blocking,
                field_group_types=field_group_types+fgroup_firstprivates
            )

            # Re-insert explicit firstprivate copies
            create_explicit_firstprivatisation(ec_phys_parallel, fprivate_map=lcopies_firstprivates)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Extracted driver routines in {s:.2f}s'):

        driver_routines = extract_driver_routines(ec_phys_parallel)
        for driver in driver_routines:
            # Create a new source file for the extracted routine
            filename = driver.name.lower() + '.F90'
            sourcefile = Sourcefile(ir=ir.Section(driver), path=build/filename)
            sourcefile.write()

            # Add an implicit C-style import to the control-flow routine
            imprt = ir.Import(module=f'{driver.name.lower()}.intfb.h', c_import=True)
            ec_phys_parallel.spec.append(imprt)

    if promote_local_arrays:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Promoted local arrays in {s:.2f}s'):
            # Bit of a hack, but easier that way
            remove_redundant_declarations(routine=ec_phys_parallel)

            promote_temporary_arrays(
                routine=ec_phys_parallel,
                horizontal=Dimension(name='horizontal', index='JL', size='YDGEOMETRY%YRDIM%NPROMA'),
                blocking=Dimension(name='blocking', index='IBL', size='YDGEOMETRY%YRDIM%NGPBLKS'),
            )

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Added OpenMP regions in {s:.2f}s'):
        # Add OpenMP pragmas around marked loops
        add_openmp_pragmas(
            routine=ec_phys_parallel,
            field_group_types=field_group_types + fgroup_dimension,
            global_variables=global_variables
        )

    # Rename DR_HOOK calls to ensure appropriate performance logging
    DrHookTransformation(
        rename={'EC_PHYS_FC': 'EC_PHYS_PARALLEL'}, kernel_only=False
    ).apply(ec_phys_parallel, role='driver')

    # Create source file, wrap as a module and write to file
    srcfile = Sourcefile(path=build/'ec_phys_parallel_mod.F90', ir=(ec_phys_parallel,))
    ModuleWrapTransformation(module_suffix='_MOD').apply(srcfile, role='kernel')

    srcfile.write()
