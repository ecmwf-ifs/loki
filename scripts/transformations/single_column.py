"""
Transformations to extract single-column Fortran code adhering to the
Single Column Abstraction (SCA), as defined by CLAW (Clement et al., 2018)
"""

from collections import OrderedDict
from loki import (
    Transformation, FindVariables, FindNodes, Transformer, SubstituteExpressions,
    SubstituteExpressionsMapper, CallStatement, Loop, Variable, Array, ArraySubscript,
    LoopRange, RangeIndex, SymbolType, DataType, fgen, as_tuple
)

__all__ = ['ExtractSCATransformation']


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
            if loop.variable == target.variable:
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
            new_dims = as_tuple(d for d, s in zip(v.dimensions.index_tuple, old_shape)
                                if s not in size_expressions)
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
                    if arg.shape is not None and len(val.dimensions.index_tuple) == 0:
                        new_dims = tuple(RangeIndex((None, None)) for _ in arg.shape)

                    # Remove target dimension sizes from caller-side argument indices
                    if val.shape is not None:
                        v_dims = val.dimensions.index_tuple or new_dims
                        new_dims = tuple(Variable(name=target.variable, scope=caller.symbols)
                                         if str(tdim).upper() in size_expressions else ddim
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
                    if str(arg).upper() == target.iteration[0]:
                        dim_lower = val
                    if str(arg).upper() == target.iteration[1]:
                        dim_upper = val

                # Remove call-side arguments (in-place)
                arguments = tuple(darg for darg, karg in zip(arguments, routine.arguments)
                                  if str(karg).upper() not in target.variables)
                kwarguments = list((darg, karg) for darg, karg in kwarguments
                                   if str(karg).upper() not in target.variables)
                new_call = call.clone(arguments=arguments, kwarguments=kwarguments)

                # Create and insert new loop over target dimension
                if wrap:
                    loop = Loop(variable=Variable(name=target.variable, scope=caller.symbols),
                                bounds=LoopRange((dim_lower, dim_upper, None)),
                                body=as_tuple([new_call]))
                    replacements[call] = loop
                else:
                    replacements[call] = new_call

        caller.body = Transformer(replacements).visit(caller.body)

        # Finally, we add the declaration of the loop variable
        if wrap and target.variable not in [str(v) for v in caller.variables]:
            # TODO: Find a better way to define raw data type
            dtype = SymbolType(DataType.INTEGER, kind='JPIM')
            caller.variables += (Variable(name=target.variable, type=dtype, scope=caller.symbols),)
