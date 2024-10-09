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

import click
from pathlib import Path
from codetiming import Timer

from loki import Sourcefile, Dimension, info
from loki.analyse import dataflow_analysis_attached
from loki.expression import symbols as sym, parse_expr
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, SubstituteExpressions,
    Transformer, CallStatement, pragmas_attached,
    pragma_regions_attached, is_loki_pragma, get_pragma_parameters
)
from loki.logging import warning
from loki.tools import as_tuple, flatten
from loki.types import DerivedType

from loki.transformations.inline import inline_marked_subroutines
from loki.transformations.sanitise import (
    merge_associates, resolve_associates,
    transform_sequence_association_append_map
)
from loki.transformations.remove_code import (
    do_remove_marked_regions, do_remove_unused_imports
)
from loki.transformations.drhook import DrHookTransformation
from loki.transformations.build_system import ModuleWrapTransformation
from loki.transformations.extract import extract_marked_subroutines, extract_subroutine
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
lcopies_firstprivates = {'ZSURF': 'ZSURF_SERIAL'}

# List of variables that we know to have global scope
global_variables = [
    'PGFL', 'PGFLT1', 'YDGSGEOM', 'YDMODEL',
    'YDDIM', 'YDSTOPH', 'YDGEOMETRY',
    'YDSURF', 'YDGMV', 'SAVTEND',
    'YGFL', 'PGMV', 'PGMVT1', 'ZGFL_DYN',
    'ZCONVCTY', 'YDDIMV', 'YDPHY2',
    'PHYS_MWAVE', 'ZSPPTGFIX', 'ZSURF_SERIAL'
]


def get_symbol(name, scope):
    """

    """
    return scope.get_symbol_scope(name).variable_map.get(name)


def substitute_spec_symbols(mapping, routine):
    """
    Do symbol substitution on the spec/declartion of a subroutine from
    a given string mapping via :any:`parse_expr`.

    Importantly, this will use the given :any:`Subroutine` as the scope.
    """
    symbol_map = {
        parse_expr(k, scope=routine): parse_expr(v, scope=routine)
        for k, v in mapping.items()
    }

    routine.spec = SubstituteExpressions(symbol_map).visit(routine.spec)

    return routine


def remove_openmp_regions(routine):
    """
    Remove any OpenMP annotations and replace with `!$loki parallel` pragmas
    """
    with pragma_regions_attached(routine):
        for region in FindNodes(ir.PragmaRegion).visit(routine.body):
            if region.pragma.keyword.lower() == 'omp':

                if 'PARALLEL' in region.pragma.content:
                    region._update(
                        pragma=ir.Pragma(keyword='loki', content='parallel'),
                        pragma_post=ir.Pragma(keyword='loki', content='end parallel')
                    )

    # Now remove all other pragmas
    pragma_map = {
        pragma: None for pragma in FindNodes(ir.Pragma).visit(routine.body)
        if pragma.keyword.lower() == 'omp'
    }
    routine.body = Transformer(pragma_map).visit(routine.body)

    # Note: This is slightly hacky, as some of the "OMP PARALLEL DO" regions
    # are not detected correctly! So instead we hook on the "OMP DO SCHEDULE"
    # and remove all other OMP pragmas.

    pragma_map = {
        pragma: None for pragma in FindNodes(ir.Pragma).visit(routine.body)
        if pragma.keyword == 'OMP'
    }
    routine.body = Transformer(pragma_map).visit(routine.body)


def add_openmp_pragmas(routine, global_variables={}, field_group_types={}):
    """
    Add the OpenMP directives for a parallel driver region with an
    outer block loop.
    """
    block_dim_size = 'YDGEOMETRY%YRDIM%NGPBLKS'

    # First get local variables and separate scalars and arrays
    routine_arguments = routine.arguments
    local_variables = tuple(
        v for v in routine.variables if v not in routine_arguments
    )
    local_scalars = tuple(
        v for v in local_variables if isinstance(v, sym.Scalar)
    )
    # Filter arrays by block-dim size, as these are global
    local_arrays = tuple(
        v for v in local_variables
        if isinstance(v, sym.Array) and not v.dimensions[-1] == block_dim_size
    )

    with pragma_regions_attached(routine):
        with dataflow_analysis_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if not is_loki_pragma(region.pragma, starts_with='parallel'):
                    return

                # Accumulate the set of locally used symbols and chase parents
                symbols = tuple(region.uses_symbols | region.defines_symbols)
                symbols = tuple(dict.fromkeys(flatten(
                    s.parents if s.parent else s for s in symbols
                )))

                # Start with loop variables and add local scalars and arrays
                local_vars = tuple(dict.fromkeys(flatten(
                    loop.variable for loop in FindNodes(ir.Loop).visit(region.body)
                )))

                local_vars += tuple(v for v in local_scalars if v in symbols)
                local_vars += tuple(v for v in local_arrays if v.name in symbols )

                # Also add used symbols that might be field groups
                local_vars += tuple(dict.fromkeys(
                    v for v in routine_arguments
                    if v.name in symbols and str(v.type.dtype) in field_group_types
                ))

                # Filter out known global variables
                local_vars = tuple(v for v in local_vars if v.name not in global_variables)

                # Make field group types firstprivate
                firstprivates = tuple(dict.fromkeys(
                    v.name for v in local_vars if v.type.dtype.name in field_group_types
                ))
                # Also make values that have an initial value firstprivate
                firstprivates += tuple(v.name for v in local_vars if v.type.initial)

                # Mark all other variables as private
                privates = tuple(dict.fromkeys(
                    v.name for v in local_vars if v.name not in firstprivates
                ))

                s_fp_vars = ", ".join(str(v) for v in firstprivates)
                s_firstprivate = f'FIRSTPRIVATE({s_fp_vars})' if firstprivates else ''
                s_private = f'PRIVATE({", ".join(str(v) for v in privates)})' if privates else ''
                pragma_parallel = ir.Pragma(
                    keyword='OMP', content=f'PARALLEL {s_private} {s_firstprivate}'
                )
                region._update(
                    pragma=pragma_parallel,
                    pragma_post=ir.Pragma(keyword='OMP', content='END PARALLEL')
                )

                # And finally mark all block-dimension loops as parallel
                with pragmas_attached(routine, node_type=ir.Loop):
                    for loop in FindNodes(ir.Loop).visit(region.body):
                        # Add OpenMP DO directives onto block loops
                        if loop.variable == 'JKGLO':
                            loop._update(
                                pragma=ir.Pragma(keyword='OMP', content='DO SCHEDULE(DYNAMIC,1)'),
                                pragma_post=ir.Pragma(keyword='OMP', content='END DO'),
                            )


def remove_block_loops(routine, field_group_types={}):
    """

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

    routine.body = RemoveBlockLoopTransformer().visit(routine.body)

    routine.body = RemoveFieldAPITransformer().visit(routine.body)


def add_block_loops(routine, field_group_types={}):
    """
    Insert IFS-style driver block-loops (NPROMA).
    """

    def _create_block_loop(body, scope):
        """
        Generate block loop object, including indexing preamble
        """
        jkglo = get_symbol('JKGLO', scope)
        icend = get_symbol('ICEND', scope)
        ibl = get_symbol('IBL', scope)

        ngptot = sym.Scalar('NGPTOT', parent=get_symbol('YDGEM', scope), scope=scope)
        nproma = sym.Scalar('NPROMA', parent=get_symbol('YDDIM', scope), scope=scope)
        lrange = sym.LoopRange((sym.Literal(1), ngptot, nproma))

        expr_tail = parse_expr('YDGEM%NGPTOT-JKGLO+1', scope=scope)
        expr_max = sym.InlineCall(
            function=sym.ProcedureSymbol('MIN', scope=scope), parameters=(nproma, expr_tail)
        )
        preamble = (ir.Assignment(lhs=icend, rhs=expr_max),)
        preamble += (ir.Assignment(
            lhs=ibl, rhs=parse_expr('(JKGLO-1)/YDDIM%NPROMA+1', scope=scope)
        ),)

        return ir.Loop(variable=jkglo, bounds=lrange, body=preamble + body)

    def _create_dim_update(scope):
        jkglo = get_symbol('JKGLO', scope)
        icend = get_symbol('ICEND', scope)
        ibl = get_symbol('IBL', scope)
        idims = get_symbol('IDIMS', scope)
        csym = sym.ProcedureSymbol(name='UPDATE', parent=idims, scope=idims.scope)
        return ir.CallStatement(name=csym, arguments=(ibl, icend, jkglo), kwarguments=())

    def _create_view_updates(section, scope):
        ibl = get_symbol('IBL', scope)
        j2d = get_symbol('J2D', scope)
        ydsppt = get_symbol('YDSPPT', scope)
        fgroup_vars = sorted(tuple(
            v for v in FindVariables(unique=True).visit(section)
            if str(v.type.dtype) in field_group_types
        ), key=lambda v: str(v))
        calls = ()
        for fgvar in fgroup_vars:
            fgsym =  get_symbol(fgvar.name, scope)
            if isinstance(fgvar, sym.Array):
                # Hacky solution for the TENDENCY_PHY corner case; inject a loop
                if not fgvar.dimensions == (j2d,):
                    continue

                csym = sym.ProcedureSymbol(name='UPDATE_VIEW', parent=fgvar, scope=fgsym.scope)
                call = (ir.CallStatement(name=csym, arguments=(ibl,), kwarguments=()),)
                bounds = sym.LoopRange(
                    (sym.Literal(1), sym.Variable(name='N2D', parent=ydsppt, scope=scope))
                )
                calls += (ir.Loop(variable=j2d, bounds=bounds, body=(call,)),)

            else:
                csym = sym.ProcedureSymbol(name='UPDATE_VIEW', parent=fgsym, scope=fgsym.scope)
                calls += (ir.CallStatement(name=csym, arguments=(ibl,), kwarguments=()),)

        return calls

    class InsertBlockLoopTransformer(Transformer):

        def visit_PragmaRegion(self, region, **kwargs):
            """
            (Re-)insert driver-level block loops into marked parallel region.
            """
            if not is_loki_pragma(region.pragma, starts_with='parallel'):
                return region

            # Filter out private copies of field group objects
            local_copies = tuple(
                a for a in FindNodes(ir.Assignment).visit(region.body)
                if isinstance(a.lhs.type.dtype, DerivedType) and \
                a.lhs.type.dtype.name in fgroup_firstprivates
            )
            idx = max(
                region.body.index(a) for a in local_copies
            ) + 1 if local_copies else 0

            # Create a block loop per marked parallel region
            scope = kwargs.get('scope')
            body = region.body[idx:]

            # Prepend FIELD API boilerplate
            preamble = (
                ir.Comment(''), ir.Comment('! Set up thread-local view pointers')
            )
            preamble += (_create_dim_update(scope),)
            preamble += _create_view_updates(body, scope)

            loop = _create_block_loop(body=preamble + body, scope=scope)

            # Add firstprivate copies before the loop
            lcopies = ()
            lvars = FindVariables(unique=True).visit(region.body)
            for lcl, gbl in lcopies_firstprivates.items():
                lhs = get_symbol(lcl, scope)
                rhs = get_symbol(gbl, scope)
                if lhs in lvars:
                    lcopies += (ir.Assignment(lhs=lhs, rhs=rhs),)

            region._update(
                body=lcopies + (ir.Comment(''), loop)
            )
            return region

    with pragma_regions_attached(routine):
        routine.body = InsertBlockLoopTransformer().visit(routine.body)


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

                call, region_routine = extract_subroutine(region, name, imports, intent_map=intent_map)

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
def inline(source, build, remove_openmp, sanitize_assoc):
    """
    Inlines EC_PHYS and CALLPAR into EC_PHYS_DRV to expose the parallel loop.
    """
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
    substitute_spec_symbols(routine=ec_phys, mapping = {
        'DIMS%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'DIMS%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
        'DIMS%KLEVS': 'YDSURF%YSP_SBD%NLEVS',
        }
    )
    substitute_spec_symbols(routine=callpar, mapping = {
        'KDIM%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'KDIM%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
        }
    )

    # Before inlining, remove DR_HOOK calls from the inner routines
    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Removed inner DR_HOOK calls in {s:.2f}s'):
        DrHookTransformation(kernel_only=False, remove=True).apply(ec_phys, role='driver')
        DrHookTransformation(kernel_only=False, remove=True).apply(callpar, role='driver')

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Inlined EC_PHYS in {s:.2f}s'):
        # First, get the outermost call
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
def parallel(source, build, remove_block_loop, promote_local_arrays):
    """
    Generate parallel regions with OpenMP and OpenACC dispatch.
    """
    source = Path(source)
    build = Path(build)

    # Get everything set up...
    ec_phys_fc = Sourcefile.from_file(source/'ec_phys_fc_mod.F90')['EC_PHYS_FC']

    # Clone original and change subroutine name
    ec_phys_parallel = ec_phys_fc.clone(name='EC_PHYS_PARALLEL')

    if remove_block_loop:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-generated block loops in {s:.2f}s'):
            # First, strip the outer block loop
            remove_block_loops(
                ec_phys_parallel, field_group_types=field_group_types+fgroup_firstprivates
            )

            # The add them back in according to parallel region
            add_block_loops(
                ec_phys_parallel, field_group_types=field_group_types+fgroup_firstprivates
            )

    if promote_local_arrays:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Promoted local arrays in {s:.2f}s'):
            # Bit of a hack, but easier that way
            remove_redundant_declarations(routine=ec_phys_parallel)

            promote_temporary_arrays(
                routine=ec_phys_parallel,
                horizontal=Dimension(name='horizontal', index='JL', size='YDGEOMETRY%YRDIM%NPROMA'),
                blocking=Dimension(name='blocking', index='IBL', size='YDGEOMETRY%YRDIM%NGPBLKS'),
            )

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
