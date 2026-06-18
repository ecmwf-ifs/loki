# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Small-kernels variant of the block-index lowering transformation.
"""

from loki.batch import ProcedureItem
from loki.ir import (
    nodes as ir, FindNodes, Transformer, pragmas_attached,
    SubstituteExpressions, FindUsedVariables, get_pragma_parameters
)
from loki.logging import warning
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType, DerivedType
from loki.expression import symbols as sym, Array, RangeIndex
from loki.transformations.block_index_transformations import LowerBlockIndexTransformation
from loki.transformations.sanitise import do_resolve_associates
from loki.transformations.utilities import (
    find_driver_loops, get_integer_variable, get_local_arrays
)

__all__ = ['LowerBlockIndexSKTransformation']


class LowerBlockIndexSKTransformation(LowerBlockIndexTransformation):
    """
    Small-kernels variant of :any:`LowerBlockIndexTransformation`.

    This transformation is pragma-driven: it only processes calls annotated
    with ``!$loki small-kernels``.  It propagates block-dimension context
    (relevant variables, driver loop template) down the call tree via
    ``item.trafo_data['LowerBlockIndex']``, enabling multi-level lowering
    through nested kernel calls.

    The transformation:

    1. Collects *relevant variables* from the driver loop header (bounds,
       step, assignment RHS variables).
    2. Passes them as new arguments to callee routines (using dtype-matching
       for derived-type kwargs).
    3. Promotes callee-local arrays with the block dimension.
    4. Injects ``!$loki unstructured-data`` pragmas around callee bodies
       for promoted locals.
    5. Updates callee argument dimensions/shapes where rank mismatch exists.
    6. Replaces block-index subscripts with range indices in call arguments.
    7. Stores transformation state in ``trafo_data['LowerBlockIndex']`` for
       successor items, enabling recursive application to nested kernels.

    .. note::

        This transformation does **not** remove the block loop from the
        driver routine.  Block-loop removal and re-injection into kernel
        routines is handled by :any:`SCCBlockSectionTransformation` and
        :any:`SCCBlockSectionToLoopTransformation`, which must follow
        this transformation in the pipeline.

    For example, the following code:

    .. code-block:: fortran

        SUBROUTINE DRIVER(NPROMA, NLEV, NB, FIELD, YDBNDS)
          USE TYPE_MOD, ONLY: BNDS_TYPE
          TYPE(BNDS_TYPE), INTENT(IN) :: YDBNDS
          REAL, INTENT(INOUT) :: FIELD(NPROMA, NLEV, NB)
          INTEGER :: IBL

          DO IBL = 1, NB
            KIDIA = YDBNDS%KIDIA
            !$loki small-kernels
            CALL KERNEL(NPROMA, NLEV, FIELD(:,:,IBL))
          END DO
        END SUBROUTINE DRIVER

        SUBROUTINE KERNEL(NPROMA, NLEV, FIELD)
          REAL, INTENT(INOUT) :: FIELD(NPROMA, NLEV)
          REAL :: TMP(NPROMA)
          ...
        END SUBROUTINE KERNEL

    is transformed to:

    .. code-block:: fortran

        SUBROUTINE DRIVER(NPROMA, NLEV, NB, FIELD, YDBNDS)
          ...
          DO IBL = 1, NB
            KIDIA = YDBNDS%KIDIA
            CALL KERNEL(NPROMA, NLEV, FIELD(:,:,:), NB=NB, KIDIA=KIDIA, &
                        YDBNDS=YDBNDS)
          END DO
        END SUBROUTINE DRIVER

        SUBROUTINE KERNEL(NPROMA, NLEV, FIELD, NB, KIDIA, YDBNDS)
          USE TYPE_MOD, ONLY: BNDS_TYPE
          REAL, INTENT(INOUT) :: FIELD(NPROMA, NLEV, NB)
          REAL :: TMP(NPROMA, NB)
          TYPE(BNDS_TYPE), INTENT(INOUT) :: YDBNDS
          !$loki unstructured-data create(TMP)
          ...
          !$loki exit unstructured-data delete(TMP)
        END SUBROUTINE KERNEL

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in
        code to define the blocking data dimension and iteration space.
    recurse_to_kernels : bool, optional
        Process kernel-role routines as well (default: ``True``).
    """

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply the small-kernels block-index lowering to *routine*.

        Dispatches to :meth:`process_driver` for driver-role routines
        and :meth:`process_kernel` for kernel-role routines (when
        ``recurse_to_kernels`` is enabled).
        """
        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))
        item = kwargs.get('item', None)
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = sub_sgraph.successors(item) if sub_sgraph is not None else ()

        do_resolve_associates(routine)

        if role == 'driver':
            self.process_driver(routine, targets, item, successors)
        elif self.recurse_to_kernels and role == 'kernel':
            self.process_kernel(routine, targets, item, successors)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_call_arg_rank(arg, block_dim_indices=None):
        """
        Retrieve the local rank of a call argument, optionally excluding
        dimensions that correspond to the block index.

        Parameters
        ----------
        arg : :any:`Variable`
            The call argument to inspect.
        block_dim_indices : tuple, optional
            Block-dimension index expressions to exclude from rank counting.
        """
        block_dim_indices = as_tuple(block_dim_indices)
        rank = len(getattr(arg, 'shape', ()))
        if getattr(arg, 'dimensions', None):
            rank = rank - len([
                d for d in arg.dimensions
                if not (isinstance(d, RangeIndex) or d in block_dim_indices)
            ])
        return rank

    def _resolve_block_dim_size(self, routine):
        """
        Resolve the block dimension size variable in the given routine.

        Iterates over ``self.block_dim.sizes`` and returns the first matching
        variable found in the routine's variable map.

        Raises
        ------
        RuntimeError
            If none of the configured size candidates can be resolved.
        """
        variable_map = routine.variable_map
        for block_dim_size in self.block_dim.sizes:
            if block_dim_size in variable_map:
                return variable_map[block_dim_size]
            if block_dim_size.split('%')[0] in variable_map:
                return get_integer_variable(routine, block_dim_size)
        raise RuntimeError(
            f'{self.__class__.__name__}: Could not resolve block dimension size '
            f'({self.block_dim.sizes}) in {routine.name}'
        )

    @staticmethod
    def _get_root_parent(var):
        """
        Walk up the parent chain of *var* and return the top-level parent.
        """
        _var = var
        while _var.parent is not None:
            _var = _var.parent
        return _var

    def _find_relevant_calls(self, routine):
        """
        Find all calls annotated with ``!$loki small-kernels`` in *routine*.

        Returns
        -------
        list of :any:`CallStatement`
        """
        relevant_calls = []
        with pragmas_attached(routine, ir.CallStatement):
            for call in FindNodes(ir.CallStatement).visit(routine.body):
                if call.pragma and any(
                    p.keyword.lower() == 'loki' and p.content.lower() == 'small-kernels'
                    for p in call.pragma
                ):
                    relevant_calls.append(call)
        return relevant_calls

    @staticmethod
    def _build_import_map(routine):
        """
        Build a mapping from symbol name (lower) to the Import node
        providing it, across all imports visible to *routine*.
        """
        import_map = {}
        for imp in routine.all_imports:
            for symbol in imp.symbols:
                import_map[symbol.name.lower()] = imp
        return import_map

    # ------------------------------------------------------------------
    # Shared per-call helpers
    # ------------------------------------------------------------------

    def _determine_new_args(self, relevant_vars, call, driver_loop_var=None):
        """
        Determine which *relevant_vars* need to be added as new arguments
        or kwargs to *call*.

        Returns
        -------
        new_args : list
            Positional arguments to add (root parents of derived-type vars
            that don't match any existing dtype, or plain scalars).
        new_kwargs : list of (str, Variable)
            Keyword arguments to add (dtype-matched derived-type parents).
        already_arg : list
            Callee argument names for vars already passed.
        """
        call_arg_map = {v: k for k, v in call.arg_map.items()}
        call_arg_dtype_map = {
            v.type.dtype: v for v in call.routine.arguments if hasattr(v, 'type')
        }

        new_args = []
        new_kwargs = []
        already_arg = []

        for var in relevant_vars:
            if driver_loop_var is not None and var == driver_loop_var:
                continue
            if isinstance(var, sym._Literal):
                continue
            if var in call_arg_map:
                already_arg.append(call_arg_map[var])
                continue

            if var.parent is not None:
                parent_var = self._get_root_parent(var)
                if parent_var not in call_arg_map:
                    dtype = parent_var.type.dtype
                    if dtype in call_arg_dtype_map:
                        new_kwargs.append((call_arg_dtype_map[dtype].name, parent_var))
                    else:
                        new_args.append(parent_var)
            else:
                new_args.append(var)

        new_kwargs = sorted(set(new_kwargs), key=lambda x: x[0].lower())
        new_args = sorted(set(new_args), key=lambda x: str(x.name))
        return new_args, new_kwargs, already_arg

    def _apply_new_args_to_call(self, call, new_args, new_kwargs, already_arg):
        """
        Update *call* with new positional/keyword arguments, add them to
        the callee's argument list, and mark already-passed args as inout.
        """
        missing_args = ()
        if new_args or new_kwargs:
            call._update(
                kwarguments=(
                    call.kwarguments
                    + as_tuple([(a.name, a) for a in new_args])
                    + as_tuple(new_kwargs)
                )
            )
            missing_args = [a for a in new_args if a not in call.routine.arguments]
            if missing_args:
                call.routine.arguments += as_tuple([
                    a.clone(scope=call.routine, type=a.type.clone(intent='inout'))
                    for a in missing_args
                ])

        if already_arg:
            for var in already_arg:
                call.routine.symbol_attrs.update({
                    var.name: call.routine.variable_map[var.name].type.clone(intent='inout')
                })

        return as_tuple(missing_args)

    @staticmethod
    def _reorder_new_dummy_declarations(routine, new_args):
        """
        Move declarations for newly-added dummy arguments into the existing
        dummy declaration block.

        ``Subroutine.arguments`` appends declarations for missing dummies to the
        end of the spec. In the small-kernels path, those new dummies can then be
        referenced in dimensions of earlier dummy-array declarations, which breaks
        some compilers. Keep the fix local to this transformation by moving the
        new declarations next to the other dummy declarations.
        """
        new_arg_names = {arg.name.lower() for arg in new_args}
        if not new_arg_names:
            return

        declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
        decl_map = {decl.symbols[0].name.lower(): decl for decl in declarations if len(decl.symbols) == 1}
        new_decls = tuple(
            decl_map[name] for name in new_arg_names
            if (name in decl_map and decl_map[name] in routine.spec.body)
        )
        if not new_decls:
            return

        existing_dummy_decls = []
        for decl in declarations:
            if decl in new_decls or decl not in routine.spec.body:
                continue
            symbols = decl.symbols
            if all(symbol.name.lower() in routine._dummies for symbol in symbols):
                existing_dummy_decls.append(decl)

        spec_body = tuple(node for node in routine.spec.body if node not in new_decls)
        insert_pos = next(
            (
                idx for idx, node in enumerate(spec_body)
                if isinstance(node, ir.VariableDeclaration)
                and not any(symbol.name.lower() in new_arg_names for symbol in node.symbols)
                and any(
                    any(part.lower() in new_arg_names for part in str(var).split('%'))
                    for var in FindUsedVariables().visit(node)
                    if hasattr(var, 'name')
                )
            ),
            None
        )
        if insert_pos is None:
            if existing_dummy_decls:
                last_dummy_decl = existing_dummy_decls[-1]
                insert_pos = spec_body.index(last_dummy_decl) + 1
            else:
                insert_pos = next(
                    (
                        idx + 1 for idx, node in enumerate(spec_body)
                        if isinstance(node, ir.Intrinsic) and node.text.upper() == 'IMPLICIT NONE'
                    ),
                    0
                )

        routine.spec = routine.spec.clone(
            body=spec_body[:insert_pos] + new_decls + spec_body[insert_pos:]
        )

    def _propagate_imports(self, new_args, call, all_import_map):
        """
        Copy derived-type and kind-parameter imports into the callee for
        any newly-added arguments.
        """
        call_imported_symbols = set()
        for imp in call.routine.all_imports:
            call_imported_symbols.update(s.name.lower() for s in imp.symbols)

        new_imports = set()
        for arg in new_args:
            if isinstance(arg.type.dtype, DerivedType):
                dtype_name = arg.type.dtype.name.lower()
                if dtype_name not in call_imported_symbols and dtype_name in all_import_map:
                    new_imports.add(all_import_map[dtype_name])
            if arg.type.kind is not None:
                kind_name = str(arg.type.kind).lower()
                if kind_name not in call_imported_symbols and kind_name in all_import_map:
                    new_imports.add(all_import_map[kind_name])

        if new_imports:
            call.routine.spec.prepend(as_tuple(new_imports))

    def promote_local_arrays(self, routine):
        """
        Promote local arrays in *routine* by appending the block dimension
        size as a trailing dimension.

        Returns
        -------
        list
            The local variables that were promoted.
        """
        local_vars = get_local_arrays(routine, routine.spec)
        local_vars = [v for v in local_vars if v.dimensions[-1] not in self.block_dim.sizes]

        if not local_vars:
            return []

        block_dim_size = self._resolve_block_dim_size(routine)
        var_map = {}
        for local_var in local_vars:
            new_dims = local_var.dimensions + (block_dim_size,)
            new_shape = local_var.shape + (block_dim_size,)
            new_type = local_var.type.clone(shape=new_shape)
            var_map[local_var] = local_var.clone(dimensions=new_dims, type=new_type)
        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)
        return local_vars

    def _inject_data_offload_pragmas(self, routine, local_vars):
        """
        Inject ``!$loki unstructured-data create/delete`` pragmas around
        the body of *routine* for the given local variables.
        """
        if not local_vars:
            return
        var_names = ', '.join(v.name for v in local_vars)

        explicit_data_vars = set()
        for pragma in FindNodes(ir.Pragma).visit(routine.body):
            keyword = pragma.keyword.lower()
            content = pragma.content.lower()
            parameters = None

            if keyword == 'acc':
                if 'enter' in content and 'data' in content:
                    parameters = get_pragma_parameters(
                        pragma, starts_with='enter data', only_loki_pragmas=False
                    )
                elif 'exit' in content and 'data' in content:
                    parameters = get_pragma_parameters(
                        pragma, starts_with='exit data', only_loki_pragmas=False
                    )
            elif keyword == 'loki':
                if 'unstructured-data' in content:
                    if content.startswith('exit unstructured-data'):
                        parameters = get_pragma_parameters(
                            pragma, starts_with='exit unstructured-data', only_loki_pragmas=False
                        )
                    else:
                        parameters = get_pragma_parameters(
                            pragma, starts_with='unstructured-data', only_loki_pragmas=False
                        )

            if not parameters:
                continue

            parameters = {
                key: ', '.join(as_tuple(value)) for key, value in parameters.items()
            }
            for category in ('create', 'delete'):
                values = parameters.get(category, '')
                if not values:
                    continue
                for entry in values.split(','):
                    entry = entry.strip()
                    if not entry:
                        continue
                    explicit_data_vars.add(entry.lower())

        if all(v.name.lower() in explicit_data_vars for v in local_vars):
            return

        pragma_start = ir.Pragma(keyword='loki', content=f'unstructured-data create({var_names})')
        pragma_end = ir.Pragma(keyword='loki', content=f'exit unstructured-data delete({var_names})')
        routine.body.prepend(pragma_start)
        routine.body.append(pragma_end)

    def _update_argument_dims(self, call):
        """
        Promote callee argument array declarations where the call-site
        argument has higher rank than the callee dummy argument.
        """
        var_map = {}
        call_variable_map = call.routine.variable_map
        block_dim_size = self._resolve_block_dim_size(call.routine)
        for arg, call_arg in call.arg_iter():
            if isinstance(arg, Array):
                if self.get_call_arg_rank(call_arg, self.block_dim.indices) > len(arg.shape):
                    callee_var = call_variable_map[arg.name]
                    new_dims = callee_var.dimensions + (block_dim_size,)
                    new_shape = callee_var.shape + (block_dim_size,)
                    new_type = callee_var.type.clone(shape=new_shape)
                    var_map[callee_var] = callee_var.clone(dimensions=new_dims, type=new_type)
        if var_map:
            call.routine.spec = SubstituteExpressions(var_map).visit(call.routine.spec)

    def _replace_block_indices_in_call(self, call):
        """
        Replace block-index subscripts with range indices ``(:)`` in call
        arguments and keyword arguments.
        """
        block_dim_indices = [idx.lower() for idx in self.block_dim.indices]

        def _process_dims(var):
            """Replace block-dim indices in var.dimensions with RangeIndex."""
            if isinstance(var, sym.Array) and var.dimensions:
                if any(str(d).lower() in block_dim_indices for d in var.dimensions):
                    new_dim = tuple(
                        sym.RangeIndex((None, None)) if str(d).lower() in block_dim_indices else d
                        for d in var.dimensions
                    )
                else:
                    new_dim = var.dimensions + (sym.RangeIndex((None, None)),)
                return var.clone(dimensions=new_dim)
            if isinstance(var, sym.Array):
                new_dim = tuple(sym.RangeIndex((None, None)) for _ in var.shape)
                return var.clone(dimensions=new_dim)
            return var

        new_arguments = tuple(_process_dims(a) for a in call.arguments)
        new_kwarguments = tuple(
            (name, _process_dims(val)) for name, val in call.kwarguments
        )
        call._update(arguments=new_arguments, kwarguments=new_kwarguments)

    def _propagate_trafo_data(self, call, relevant_vars, successor_map,
                              driver_loop=None, item=None):
        """
        Store transformation state in the successor's ``trafo_data`` so
        that nested kernels can continue the lowering.

        Parameters
        ----------
        call : :any:`CallStatement`
            The call whose target routine is the successor.
        relevant_vars : tuple
            Variables to propagate (will be substituted via call arg map).
        successor_map : :any:`CaseInsensitiveDict`
            Map from routine local_name to successor item.
        driver_loop : :any:`Loop`, optional
            The driver loop template to propagate (for kernel-level reuse).
        item : :any:`ProcedureItem`, optional
            The current item (used to read existing driver_loop from trafo_data).
        """
        call_name = str(call.name).lower()
        if call_name not in successor_map:
            return

        successor = successor_map[call_name]
        successor.trafo_data.setdefault('LowerBlockIndex', {})
        successor.trafo_data['LowerBlockIndex'].setdefault('assignments', {})

        call_arg_map = {v: k for k, v in call.arg_map.items()}

        if 'relevant_vars' not in successor.trafo_data['LowerBlockIndex']:
            successor.trafo_data['LowerBlockIndex']['relevant_vars'] = (
                SubstituteExpressions(call_arg_map).visit(as_tuple(relevant_vars))
            )

        # Determine the driver loop to propagate
        if driver_loop is not None:
            # Driver path: clone loop, strip calls and pragmas, substitute args
            drv_loop = driver_loop.clone()
            drv_loop = Transformer(
                {c: None for c in FindNodes(ir.CallStatement).visit(drv_loop.body)}
            ).visit(drv_loop)
            drv_loop = Transformer(
                {p: None for p in FindNodes(ir.Pragma).visit(drv_loop.body)}
            ).visit(drv_loop)
            drv_loop = SubstituteExpressions(call_arg_map).visit(drv_loop)
        elif item is not None and 'LowerBlockIndex' in item.trafo_data:
            # Kernel path: re-use parent's driver loop, substituted
            drv_loop = item.trafo_data['LowerBlockIndex'].get('driver_loop')
            if drv_loop is not None:
                drv_loop = SubstituteExpressions(call_arg_map).visit(drv_loop)
        else:
            drv_loop = None

        if drv_loop is not None and 'driver_loop' not in successor.trafo_data['LowerBlockIndex']:
            successor.trafo_data['LowerBlockIndex']['driver_loop'] = drv_loop

    # ------------------------------------------------------------------
    # Main dispatch methods
    # ------------------------------------------------------------------

    def process_driver(self, routine, targets, item, successors):
        """
        Process a driver-role routine: find driver loops containing calls
        annotated with ``!$loki small-kernels``, collect relevant variables
        from loop headers, and apply the block-index lowering to each call.
        """
        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor) for successor in successors
        )
        all_import_map = self._build_import_map(routine)

        # Find driver loops and pragma-annotated calls within them
        with pragmas_attached(routine, ir.Loop):
            driver_loops = find_driver_loops(routine.body, targets)

        relevant_calls = []
        with pragmas_attached(routine, ir.CallStatement):
            for driver_loop in driver_loops:
                for call in FindNodes(ir.CallStatement).visit(driver_loop.body):
                    if call.pragma and any(
                        p.keyword.lower() == 'loki' and p.content.lower() == 'small-kernels'
                        for p in call.pragma
                    ):
                        relevant_calls.append((call, driver_loop))

        # Collect assignments in each driver loop body (used to extract
        # relevant variables via FindUsedVariables)
        driver_loop_assignments = {}
        for _, driver_loop in relevant_calls:
            if driver_loop not in driver_loop_assignments:
                driver_loop_assignments[driver_loop] = (
                    FindNodes(ir.Assignment).visit(driver_loop.body)
                )

        for call, driver_loop in relevant_calls:
            if str(call.name).lower() not in targets:
                continue
            if call.routine is BasicType.DEFERRED:
                warning(
                    f'[{self.__class__.__name__}] Not processing routine '
                    f'{call.name}. Call statement not enriched'
                )
                continue

            # Collect relevant variables from loop header
            relevant_vars = set()
            relevant_vars.add(driver_loop.bounds.lower)
            relevant_vars.add(driver_loop.bounds.upper)
            if driver_loop.bounds.step is not None:
                relevant_vars.add(driver_loop.bounds.step)
            relevant_vars.update(
                FindUsedVariables().visit(driver_loop_assignments[driver_loop])
            )

            # Compute and apply new arguments
            new_args, new_kwargs, already_arg = self._determine_new_args(
                relevant_vars, call, driver_loop_var=driver_loop.variable
            )
            missing_args = self._apply_new_args_to_call(call, new_args, new_kwargs, already_arg)

            # Ensure driver loop variable is declared in callee
            if driver_loop.variable not in call.routine.variable_map:
                call.routine.variables += (driver_loop.variable.clone(scope=call.routine),)

            # Propagate imports for new args
            self._propagate_imports(new_args, call, all_import_map)

            # Promote local arrays and inject data pragmas
            local_vars = self.promote_local_arrays(call.routine)
            self._inject_data_offload_pragmas(call.routine, local_vars)

            # Update argument dimensions where rank mismatch
            self._update_argument_dims(call)
            self._reorder_new_dummy_declarations(call.routine, missing_args)

            # Replace block indices with range indices in call args
            self._replace_block_indices_in_call(call)

            # Propagate trafo_data to successors
            self._propagate_trafo_data(
                call, relevant_vars, successor_map,
                driver_loop=driver_loop, item=item
            )

    def process_kernel(self, routine, targets, item, successors):
        """
        Process a kernel-role routine: read ``relevant_vars`` from
        ``item.trafo_data['LowerBlockIndex']`` (set by the parent caller)
        and apply the same block-index lowering to nested calls annotated
        with ``!$loki small-kernels``.
        """
        if 'LowerBlockIndex' not in item.trafo_data:
            return
        relevant_vars = item.trafo_data['LowerBlockIndex'].get('relevant_vars', ())
        if not relevant_vars:
            return

        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor) for successor in successors
        )
        all_import_map = self._build_import_map(routine)

        relevant_calls = self._find_relevant_calls(routine)
        if not relevant_calls:
            return

        for call in relevant_calls:
            if str(call.name).lower() not in targets:
                continue
            if call.routine is BasicType.DEFERRED:
                warning(
                    f'[{self.__class__.__name__}] Not processing routine '
                    f'{call.name}. Call statement not enriched'
                )
                continue

            # Compute and apply new arguments
            new_args, new_kwargs, already_arg = self._determine_new_args(
                relevant_vars, call
            )
            missing_args = self._apply_new_args_to_call(call, new_args, new_kwargs, already_arg)

            # Propagate imports for new args
            self._propagate_imports(new_args, call, all_import_map)

            # Promote local arrays and inject data pragmas
            local_vars = self.promote_local_arrays(call.routine)
            self._inject_data_offload_pragmas(call.routine, local_vars)

            # Update argument dimensions where rank mismatch
            self._update_argument_dims(call)
            self._reorder_new_dummy_declarations(call.routine, missing_args)

            # Replace block indices with range indices in call args
            self._replace_block_indices_in_call(call)

            # Propagate trafo_data to successors
            self._propagate_trafo_data(
                call, relevant_vars, successor_map, item=item
            )
