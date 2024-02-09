from loki import (
     Transformation, FindNodes, ir, FindScopes, as_tuple, flatten, Transformer,
     NestedTransformer, FindVariables, demote_variables, is_dimension_constant,
     is_loki_pragma, dataflow_analysis_attached, BasicType, pragmas_attached,
     SubstituteExpressions, symbols as sym
)
from transformations.single_column_coalesced import SCCBaseTransformation

__all__ = [
    'SCCLowerLoopTransformation'
]

class SCCLowerLoopTransformation(Transformation):
    """
    ...

    PROBLEM FOR KERNELS CALLED WITHIN DIFFERENT LOOPS WITH DIFFERENT ARGUMENTS
    WHICH WOULD MAKE IT NECESSARY TO DUPLICATE ?! THE KERNEL ...

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, dimension, dim_name=None):
        self.dimension = dimension
        # if loop_index is None:
        if dim_name is None:
            # self.loop_index = dimension.index
            self.dim_name = dimension.index
        else:
            # self.loop_index = loop_index
            self.dim_name = dim_name

    @classmethod
    def some_utility_method(cls, section, routine, horizontal):
        """
        ...

        Parameters
        ----------
        section : tuple of :any:`Node`
            A section of nodes to be wrapped in a vector-level loop
        routine : :any:`Subroutine`
            The subroutine in the vector loops should be removed.
        horizontal: :any:`Dimension`
            The dimension specifying the horizontal vector dimension
        """

        ...

    def transform_subroutine(self, routine, **kwargs):
        """
        ...

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """
        role = kwargs['role']
        
        SCCBaseTransformation.explicit_dimensions(routine)

        if role == 'kernel':
            self.process_kernel(routine)
        if role == 'driver':
            self.process_driver(routine)

        SCCBaseTransformation.remove_dimensions(routine)

    def _promote_locals(self, routine, index=-1):
        var_map = {}
        for var in routine.variables:
            if isinstance(var, sym.Array) and var not in routine.arguments:
                var_shape = list(var.shape)
                var_shape.insert(index, routine.variable_map[self.dimension.size]) 
                var_dimensions = list(var.dimensions)
                var_dimensions.insert(index, routine.variable_map[self.dimension.size])
                var_map[var] = var.clone(type=var.type.clone(shape=as_tuple(var_shape)), dimensions=as_tuple(var_dimensions))
        
        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)
        var_map = {}
        for var in FindVariables(unique=False).visit(routine.body):
            if isinstance(var, sym.Array) and var.name not in [arg.name for arg in routine.arguments]:
                if not var.dimensions:
                    continue
                var_shape = list(var.shape)
                var_shape.insert(index, routine.variable_map[self.dimension.size])
                var_dimensions = list(var.dimensions)
                var_dimensions.insert(index, routine.variable_map[self.dim_name])
                # var_map[var] = var.clone(type=var.type.clone(shape=as_tuple(var_shape)), dimensions=as_tuple(var_dimensions))
                var_map[var] = var.clone(dimensions=as_tuple(var_dimensions))

        routine.body = SubstituteExpressions(var_map).visit(routine.body)

    def _insert_loop_in_kernel(self, routine, call, loop):
        kernel_loops = FindNodes(ir.Loop).visit(call.routine.body)
        if not any(kernel_loop.variable == loop.variable  for kernel_loop in kernel_loops):
           
            dim_name_assignment = ir.Comment(text="")
            loop_variable = routine.variable_map[loop.variable.name]
            loop_variable_type = loop_variable.type.clone(intent=None)
            call.routine.variables += (loop_variable.clone(scope=call.routine, type=loop_variable_type),)
            if self.dim_name != loop_variable.name:
                dim_variable = routine.variable_map[self.dim_name]
                dim_variable_type = dim_variable.type.clone(intent=None)
                call.routine.variables += (dim_variable.clone(scope=call.routine, type=dim_variable_type),)
                assignments = FindNodes(ir.Assignment).visit(loop.body)
                # TODO: maybe not all the used variables within the assignment within kernel available ...
                for assignment in assignments:
                    if assignment.lhs == self.dim_name:
                        dim_name_assignment = assignment.clone() 

            # TODO: args vs. kwargs ...
            # loop_start = routine.variable_map[loop.bounds[0].name].clone(name=f"{loop.variable.name}_start")
            # loop_var_type = loop_start.type.clone(intent='in')
            loop_arg_type = loop_variable_type.clone(intent='in')
            loop_start = loop_variable.clone(name=f"{loop.variable.name}_start", type=loop_arg_type)
            loop_end = loop_start.clone(name=f"{loop.variable.name}_end")
            loop_step = loop_start.clone(name=f"{loop.variable.name}_step")
            # call.routine.kwarguments += ((loop_start.name, loop_start), (loop_end.name, loop_end), (loop_step.name, loop_step)) 
            call.routine.arguments += (loop_start, loop_end, loop_step, routine.variable_map[self.dimension.size])

            call.routine.body = (ir.Comment(text=""), 
                    ir.Comment(text=f"! START of Loki inserted loop {loop.variable}"),
                    loop.clone(bounds=sym.Range((call.routine.variable_map[f"{loop.variable.name}_start"],
                        call.routine.variable_map[f"{loop.variable.name}_end"],
                        call.routine.variable_map[f"{loop.variable.name}_step"])), body=(dim_name_assignment, call.routine.body,)),
                    ir.Comment(text=f"! END of Loki inserted loop {loop.variable}"))
        
        start = loop.bounds.children[0]
        end = loop.bounds.children[1]
        step = loop.bounds.children[2] if len(loop.bounds.children) > 2 else sym.IntLiteral(1)
        return ((f"{loop.variable.name}_start", start), (f"{loop.variable.name}_end", end),
                (f"{loop.variable.name}_step", step), (routine.variable_map[self.dimension.size].name, routine.variable_map[self.dimension.size].clone(scope=call.routine)))

    def process_driver(self, routine):
        # if DRIVER
        #  find loops with relevant loop index
        #   remove loop, but leave some Loki pragma (so the information is not lost for e.g. launch configuration ...) ?!
        #  find calls and arguments with relevant dimension
        #   remove dimension
        #    either here or separate/independent trafo: convert arr(:, :, :) to arr
        #      if this is not possible throw error if necessary (e.g. for CUF/"C" ...)
        loops = FindNodes(ir.Loop).visit(routine.body)
        loops = [loop for loop in loops if loop.variable == self.dimension.index]
        loop_map = {}
        comment = ir.Comment(text="")
        relevant_callees = []
        for loop in loops:
            # driver loop mapping
            loop_map[loop] = (comment, ir.Pragma(keyword="loki", content="start: removed loop"),
                    comment, loop.body, comment, ir.Pragma(keyword="loki", content="end: removed loop"), comment)
            # calls
            calls = FindNodes(ir.CallStatement).visit(loop.body)
            # call_map
            call_routine_map = {}
            for call in calls:
                relevant_callees.append(call.routine)
                call_routine_map[call.routine.name] = {}
                call_routine_map[call.routine.name]['shape'] = {}
                call_routine_map[call.routine.name]['dim'] = {}
                # call_map[call] = call.clone(kwarguments=call.kwarguments + self._insert_loop_in_kernel(routine, call, loop))
                new_kwarguments = self._insert_loop_in_kernel(routine, call, loop)
                new_arguments = ()
                for i_arg, arg in enumerate(call.arguments): # TODO: should use .arg_iter() ...
                    if isinstance(arg, sym.Array):
                        replace_indices = [dim == self.dim_name for dim in arg.dimensions]
                        # call.routine
                        if any(replace_indices):
                            relevant_index = 0
                            for i in range(len(replace_indices)):
                                if replace_indices[i]:
                                    relevant_index = i
                            corresponding_var = call.routine.arguments[i_arg]
                            arg_shape = arg.shape
                            call_arg_shape = list(call.routine.variable_map[corresponding_var.name].shape)
                            # call_arg_shape.insert(relevant_index, arg_shape[relevant_index])
                            call_arg_dim = list(call.routine.variable_map[corresponding_var.name].dimensions)
                            # call_arg_dim.insert(relevant_index, call.routine.variable_map[loop.variable.name])
                            call_routine_map[call.routine.name]['shape'][call.routine.variable_map[corresponding_var.name]] = (relevant_index, arg_shape[relevant_index]) # call.routine.variable_map[corresponding_var.name].shape[relevant_index])
                            if loop.variable.name != self.dim_name:
                                call_routine_map[call.routine.name]['dim'][corresponding_var.name] = (relevant_index, self.dim_name)
                            else:
                                call_routine_map[call.routine.name]['dim'][corresponding_var.name] = (relevant_index, loop.variable.name)
                            # call_routine_map[call.routine.name]['dim'][call.routine.variable_map[corresponding_var.name]] = (relevant_index, loop.variable.name) # call.routine.variable_map[loop.variable.name])
                            # call_routine_map[call.routine][call.routine.variable_map[corresponding_var.name]] = call.routine.variable_map[corresponding_var.name].clone(shape=)
                        # call
                        new_dimensions = [dim if dim != self.dim_name else sym.RangeIndex((None, None)) for dim in arg.dimensions]
                        new_arguments += (arg.clone(dimensions=as_tuple(new_dimensions)),)
                    else:
                        new_arguments += (arg,)
                call._update(arguments=new_arguments, kwarguments=call.kwarguments+new_kwarguments)
        # call.routine.body = (loop.clone(body=(call.routine.body,)),)
        routine.body = Transformer(loop_map).visit(routine.body)

        relevant_callees = set(relevant_callees)
        for relevant_callee in relevant_callees:

            self._promote_locals(relevant_callee)

            shape_map = {}
            for key, elem in call_routine_map[call.routine.name]['shape'].items():
                var_dims = list(key.dimensions)
                var_dims.insert(elem[0], elem[1])
                shape_map[key] = key.clone(type=key.type.clone(shape=as_tuple(list(key.shape).insert(elem[0], elem[1]))), dimensions=as_tuple(var_dims))# dimensions=as_tuple(list(key.dimensions).insert(elem[0], elem[1])))
            relevant_callee.spec = SubstituteExpressions(shape_map).visit(relevant_callee.spec)
            dim_map = {}
            relevant_variables = [key for key in call_routine_map[call.routine.name]['dim']] # key.name ?
            variables = FindVariables(unique=True).visit(relevant_callee.body)
            for var in variables:
                if var.name in relevant_variables:
                    if not var.dimensions:
                        continue
                    var_dims = list(var.dimensions)
                    var_dims.insert(call_routine_map[call.routine.name]['dim'][var.name][0], relevant_callee.variable_map[call_routine_map[call.routine.name]['dim'][var.name][1]])
                    dim_map[var] = var.clone(dimensions=as_tuple(var_dims))
            relevant_callee.body = SubstituteExpressions(dim_map).visit(relevant_callee.body)

    def process_kernel(self, routine):
        # if KERNEL
        # recreate loop
        # introdcue dimension (for relevant arrays)
        ...
