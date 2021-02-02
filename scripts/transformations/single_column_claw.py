"""
Transformations to extract single-column Fortran code adhering to the
Single Column Abstraction (SCA), as defined by CLAW (Clement et al., 2018)
"""

from collections import OrderedDict
from loki import (
    Transformation, FindVariables, FindNodes, Transformer, SubstituteExpressions,
    SubstituteExpressionsMapper, Assignment, CallStatement, Loop, Variable,
    Array, Pragma, Declaration, ArraySubscript, LoopRange, RangeIndex,
    SymbolType, BasicType, CaseInsensitiveDict, as_tuple, warning
)


__all__ = ['ExtractSCATransformation', 'CLAWTransformation']


class ExtractSCATransformation(Transformation):
    """
    Transformation to transform vectorized Frotran kernel into SCA format.

    Note, this requires preprocessing with the `DerivedTypeArgumentsTransformation`.
    """

    def __init__(self, dimension):
        self.dimension = dimension

    def transform_subroutine(self, routine, **kwargs):
        task = kwargs.get('task', None)
        role = kwargs['role'] if task is None else task.config['role']

        if role == 'driver':
            self.hoist_dimension_from_call(routine, target=self.dimension, wrap=True)

        elif role == 'kernel':
            self.hoist_dimension_from_call(routine, target=self.dimension, wrap=False)
            self.remove_dimension(routine, target=self.dimension)

        if routine.members is not None:
            for member in routine.members:
                self.apply(member, **kwargs)

    @staticmethod
    def remove_dimension(routine, target):
        """
        Remove all loops and variable indices of a given target dimension
        from the given routine.
        """
        size_expressions = target.size_expressions

        # Remove all loops over the target dimensions
        loop_map = OrderedDict()
        for loop in FindNodes(Loop).visit(routine.body):
            if loop.variable == target.index:
                loop_map[loop] = loop.body

        routine.body = Transformer(loop_map).visit(routine.body)

        # Drop declarations for dimension variables (eg. loop counter or sizes)
        # Note that this also removes arguments and their declarations!
        routine.variables = [v for v in routine.variables if v not in target.variables]

        # Establish the new dimensions and shapes first, before cloning the variables
        # The reason for this is that shapes of all variable instances are linked
        # via caching, meaning we can easily void the shape of an unprocessed variable.
        variables = list(routine.variables)
        variables += list(FindVariables(unique=False).visit(routine.body))

        # We also include the member routines in the replacement process, as they share
        # declarations.
        for m in as_tuple(routine.members):
            variables += list(FindVariables(unique=False).visit(m.body))
        variables = [v for v in variables if isinstance(v, Array) and v.shape is not None]
        shape_map = {v.name: v.shape for v in variables}

        # Now generate a mapping of old to new variable symbols
        vmap = {}
        for v in variables:
            old_shape = shape_map[v.name]
            new_shape = as_tuple(s for s in old_shape if s not in size_expressions)
            if v.dimensions:
                new_dims = as_tuple(d for d, s in zip(v.dimensions.index_tuple, old_shape)
                                    if s not in size_expressions)
            else:
                new_dims = ()
            new_dims = None if len(new_dims) == 0 else ArraySubscript(new_dims)
            if len(old_shape) != len(new_shape):
                new_type = v.type.clone(shape=new_shape)
                vmap[v] = v.clone(dimensions=new_dims, type=new_type)

        # Apply vmap to variable and argument list and subroutine body
        routine.variables = [vmap.get(v, v) for v in routine.variables]

        # Apply substitution map to replacements to capture nesting
        mapper = SubstituteExpressionsMapper(vmap)
        vmap2 = {k: mapper(v) for k, v in vmap.items()}

        routine.body = SubstituteExpressions(vmap2).visit(routine.body)
        for m in as_tuple(routine.members):
            m.body = SubstituteExpressions(vmap2).visit(m.body)

    @staticmethod
    def hoist_dimension_from_call(caller, target, wrap=True):
        """
        Remove all indices and variables of a target dimension from
        caller (driver) and callee (kernel) routines, and insert the
        necessary loop over the target dimension into the driver.

        Note: In order for this routine to see the target dimensions
        in the argument declarations of the kernel, it must be applied
        before they are stripped from the kernel itself.
        """
        size_expressions = target.size_expressions
        replacements = {}

        for call in FindNodes(CallStatement).visit(caller.body):
            if call.context is not None and call.context.active:
                routine = call.context.routine
                argmap = {}

                # Replace target dimension with a loop index in arguments
                for arg, val in call.context.arg_iter(call):
                    if not isinstance(arg, Array) or not isinstance(val, Array):
                        continue

                    # TODO: Properly construct the vmap with updated dims for the call
                    new_dims = None

                    # Insert ':' for all missing dimensions in argument
                    if arg.shape is not None and not val.dimensions:
                        new_dims = tuple(RangeIndex((None, None)) for _ in arg.shape)

                    # Remove target dimension sizes from caller-side argument indices
                    if val.shape is not None:
                        v_dims = val.dimensions.index_tuple if val.dimensions else new_dims
                        new_dims = tuple(Variable(name=target.index, scope=caller.scope)
                                         if tdim in size_expressions else ddim
                                         for ddim, tdim in zip(v_dims, val.shape))

                    if new_dims is not None:
                        argmap[val] = val.clone(dimensions=ArraySubscript(new_dims))

                # Apply argmap to the list of call arguments
                arguments = [argmap.get(a, a) for a in call.arguments]
                kwarguments = as_tuple((k, argmap.get(a, a)) for k, a in call.kwarguments)

                # Collect caller-side expressions for dimension sizes and bounds
                dim_lower = None
                dim_upper = None
                for arg, val in call.context.arg_iter(call):
                    if arg == target.bounds[0]:
                        dim_lower = val
                    if arg == target.bounds[1]:
                        dim_upper = val

                # Remove call-side arguments (in-place)
                arguments = tuple(darg for darg, karg in zip(arguments, routine.arguments)
                                  if karg not in target.variables)
                kwarguments = list((darg, karg) for darg, karg in kwarguments
                                   if karg not in target.variables)
                new_call = call.clone(arguments=arguments, kwarguments=kwarguments)

                # Create and insert new loop over target dimension
                if wrap:
                    loop = Loop(variable=Variable(name=target.index, scope=caller.scope),
                                bounds=LoopRange((dim_lower, dim_upper, None)),
                                body=as_tuple([new_call]))
                    replacements[call] = loop
                else:
                    replacements[call] = new_call

        caller.body = Transformer(replacements).visit(caller.body)

        # Finally, we add the declaration of the loop variable
        if wrap and target.index not in [str(v) for v in caller.variables]:
            # TODO: Find a better way to define raw data type
            dtype = SymbolType(BasicType.INTEGER, kind='JPIM')
            caller.variables += (Variable(name=target.index, type=dtype, scope=caller.scope),)


class CLAWTransformation(ExtractSCATransformation):
    """
    Transformation to extract SCA Fortran and apply the necessary CLAW annotations.

    Note, this requires preprocessing with the `DerivedTypeArgumentsTransformation`.

    :param claw_data_offload: Flag triggering the insert of CLAW data offload regions
                              (via OpenACC ``create`` and ``update`` pragmas).
    """

    def __init__(self, **kwargs):
        self.claw_data_offload = kwargs.pop('claw_data_offload', True)
        super().__init__(**kwargs)

        # We need to keep track of the depth of items in the tree
        self.item_depth = CaseInsensitiveDict()

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role')
        targets = as_tuple(kwargs.get('targets', None))
        if targets:
            targets = tuple(t.lower() for t in targets)

        if role == 'driver':
            self.item_depth[routine.name.lower()] = 0

        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name.lower() in targets:
                if call.context:
                    self.item_depth[call.name.lower()] = self.item_depth[routine.name.lower()] + 1
                else:
                    warning('[Loki] CLAWTransform: Routine {} not attached to call context in {}'.format(
                        routine.name, call.name.lower()
                    ))

        # Store the names of all variables that we are about to remove
        claw_vars = [v.name for v in routine.variables
                     if isinstance(v, Array) and v.shape[0] in self.dimension.size_expressions]

        # The CLAW assumes that variables defining dimension sizes or iteration spaces
        # exist in both driver and kernel as local variables. We often rely on implicit
        # association in the function call though, so we generate local version of the
        # dimension variables in the calling driver routine before applying SCA.
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.context is not None and call.context.active:

                # Explicitly create and assign local variables
                # that mimic dimension variables in the kernel
                assignments = []
                for arg, val in call.context.arg_iter(call):
                    if arg == self.dimension.size and not arg.name in routine.variables:
                        local_var = arg.clone(scope=routine.scope, type=arg.type.clone(intent=None))
                        assignments.append(Assignment(lhs=local_var, rhs=val))
                        routine.spec.append(Declaration(variables=[local_var]))

                    if arg == self.dimension.bounds[0] and not arg.name in routine.variables:
                        local_var = arg.clone(scope=routine.scope, type=arg.type.clone(intent=None))
                        assignments.append(Assignment(lhs=local_var, rhs=val))
                        routine.spec.append(Declaration(variables=[local_var]))

                    if arg == self.dimension.bounds[1] and not arg.name in routine.variables:
                        local_var = arg.clone(scope=routine.scope, type=arg.type.clone(intent=None))
                        assignments.append(Assignment(lhs=local_var, rhs=val))
                        routine.spec.append(Declaration(variables=[local_var]))

                routine.body = Transformer({call: assignments + [call]}).visit(routine.body)

        # Invoke the actual SCA format extraction
        super().transform_subroutine(routine, **kwargs)

        if role == 'kernel':
            # Gather all declarations for variables that have been demoted during SCA
            declarations = FindNodes(Declaration).visit(routine.spec)
            decl_map = dict((v, decl) for decl in declarations for v in decl.variables)
            claw_decls = [decl for v, decl in decl_map.items() if v.name in claw_vars]

            # Remove declarations from spec temporarily
            routine.spec = Transformer({decl: None for decl in claw_decls}).visit(routine.spec)

            # Create CLAW declarations and mark with `!$claw model-data` pragmas
            claw_decls = [Pragma(keyword='claw', content='model-data')] + claw_decls
            claw_decls += [Pragma(keyword='claw', content='end model-data')]
            routine.spec.append(claw_decls)

            # Add the `!$claw sca` and `!$claw sca routine` pragmas
            rname = routine.name.lower()
            if rname in self.item_depth and self.item_depth[rname] == 1:
                routine.spec.append([Pragma(keyword='claw', content='sca')])
            else:
                routine.spec.append([Pragma(keyword='claw', content='sca routine')])

            # Insert `!$claw sca forward` pragmas to propagate the SCA region
            for call in FindNodes(CallStatement).visit(routine.body):
                if call.name.lower() in targets:
                    call._update(pragma=Pragma(keyword='claw', content='sca forward'))

        if role == 'driver':
            # Insert loop pragmas in driver (in-place)
            for loop in FindNodes(Loop).visit(routine.body):
                claw_keywords = 'sca forward'
                if self.claw_data_offload:
                    claw_keywords += ' create update'

                if loop.variable == self.dimension.index:
                    pragma = Pragma(keyword='claw', content=claw_keywords)
                    loop._update(pragma=pragma)
