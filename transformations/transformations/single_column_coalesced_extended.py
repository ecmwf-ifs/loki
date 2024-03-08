from itertools import chain
from loki import (
     Transformation, FindNodes, ir, FindScopes, as_tuple, flatten, Transformer,
     NestedTransformer, FindVariables, demote_variables, is_dimension_constant,
     is_loki_pragma, dataflow_analysis_attached, BasicType, pragmas_attached,
     SubstituteExpressions, symbols as sym, fgen
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

    def __init__(self, dimension, dim_name=None, keep_driver_loop=False, ignore_dim_name=False):
        self.dimension = dimension
        self.driver_dim_name = None
        # if loop_index is None:
        if dim_name is None or ignore_dim_name:
            # self.loop_index = dimension.index
            self.dim_name = dimension.index
            self.driver_dim_name = dim_name
        else:
            # self.loop_index = loop_index
            self.dim_name = dim_name
        # TODO: this is not really a clean solution, as the resulting code
        #  is not valid if keep_driver_loop=True, however, this
        #  facilitates further processing ...
        self.keep_driver_loop = keep_driver_loop
        self.ignore_dim_name = ignore_dim_name
        self.call_routines = []

    def transform_subroutine(self, routine, **kwargs):
        """
        ...

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """
        role = kwargs['role']
        item = kwargs.get('item', None)
        depths = kwargs.get('depths', None)
        targets = kwargs.get('targets', None)
        if depths is None:
            if role == 'driver':
                depth = 0
            elif role == 'kernel':
                depth = 1
        else:
            depth = depths[item]

        SCCBaseTransformation.explicit_dimensions(routine)

        self.process_routine(routine, targets, role, remove_loop=role=='driver',
                insert_index_instead_of_loop=role!='driver')

        SCCBaseTransformation.remove_dimensions(routine, calls_only=True)

    @staticmethod
    def _remove_vector_sections(routine):
        section_mapper = {s: s.body for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section'}
        if section_mapper:
            routine.body = Transformer(section_mapper).visit(routine.body)

    def _promote_locals(self, routine, index=None):
        """
        Promote locals/temporaries with self.dimension ...

        if ``index=None``, it will be appended, otherwise inserted
        at ``ìndex``.
        """
        var_map = {}
        for var in routine.variables:
            if isinstance(var, sym.Array) and var not in routine.arguments:
                var_shape = list(var.shape)
                var_dimensions = list(var.dimensions)
                if index is None:
                    var_shape.append(SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)) # routine.variable_map[self.dimension.size])
                    var_dimensions.append(SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)) # routine.variable_map[self.dimension.size])
                else:
                    var_shape.insert(index, SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)) # routine.variable_map[self.dimension.size]) 
                    var_dimensions.insert(index, SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)) # routine.variable_map[self.dimension.size])
                var_map[var] = var.clone(type=var.type.clone(shape=as_tuple(var_shape)), dimensions=as_tuple(var_dimensions))
        
        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)
        var_map = {}
        for var in FindVariables(unique=False).visit(routine.body):
            if isinstance(var, sym.Array) and var.name not in [arg.name for arg in routine.arguments]:
                if not var.dimensions or not var.shape:
                    continue
                var_shape = list(var.shape)
                var_dimensions = list(var.dimensions)
                if index is None:
                    var_shape.append(SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)) # routine.variable_map[self.dimension.size])
                    var_dimensions.append(routine.variable_map[self.dim_name])
                else:
                    var_shape.insert(index, SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)) # routine.variable_map[self.dimension.size])
                    var_dimensions.insert(index, routine.variable_map[self.dim_name])
                
                # TODO: which one?
                var_map[var] = var.clone(type=var.type.clone(shape=as_tuple(var_shape)), dimensions=as_tuple(var_dimensions))
                # var_map[var] = var.clone(dimensions=as_tuple(var_dimensions))

        routine.body = SubstituteExpressions(var_map).visit(routine.body)

    def _insert_index_in_kernel(self, routine, call, loop=None):
        index_variable = routine.variable_map[self.dim_name]
        if index_variable.name not in call.routine.arguments:
            dimension_size_var = SCCBaseTransformation.get_integer_variable(call.routine, self.dimension.size)
            call.routine.arguments += (index_variable.clone(scope=call.routine, type=index_variable.type.clone(intent='in')),
                    dimension_size_var.clone(type=dimension_size_var.type.clone(intent='in'), scope=call.routine)) # SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)) # routine.variable_map[self.dimension.size])
        return ((index_variable.name, index_variable), (self.dimension.size, SCCBaseTransformation.get_integer_variable(routine, self.dimension.size))) # (routine.variable_map[self.dimension.size].name, routine.variable_map[self.dimension.size]))

    def _insert_loop_in_kernel(self, routine, call, loop):
        """
        insert loop in kernel ...
        """
        # TODO: do not insert loop, but pass loop index/variable ?!
        kernel_loops = FindNodes(ir.Loop).visit(call.routine.body)
        # possible assignment within loop (if relevant dimension and loop variable are different)
        dim_name_assignment = ir.Comment(text="")
        # possible additional arguments for assignment within loop
        additional_arguments = ()
        additional_variables = ()
        # the loop variable
        loop_variable = routine.variable_map[loop.variable.name]
        loop_variable_type = loop_variable.type.clone(intent=None)

        # if relevant dimension and loop variable are different: add assignment and relevant variables
        if self.dim_name != loop_variable.name: #  and not self.ignore_dim_name:
            dim_variable = routine.variable_map[self.dim_name]
            dim_variable_type = dim_variable.type.clone(intent=None)
            # TODO: this could be called several times for 
            # call.routine.variables += (dim_variable.clone(scope=call.routine, type=dim_variable_type),)
            additional_variables += (dim_variable.clone(scope=call.routine, type=dim_variable_type),)
            if not self.ignore_dim_name:
                assignments = FindNodes(ir.Assignment).visit(loop.body)
                # DONE: maybe not all the used variables within the assignment within kernel available ...
                # TODO: and this one can be different for differnt loops/kernel calls ...
                for assignment in assignments:
                    if assignment.lhs == self.dim_name:
                        dim_name_assignment = assignment.clone()
                        for var in FindVariables().visit(assignment.rhs):
                            if var not in [loop.variable, self.dim_name, self.dimension.index, self.dimension._aliases]:
                                additional_arguments += (var,)
                # remove duplicates in additional_arguments ...
                additional_arguments = sorted(list(set(additional_arguments)), key=lambda symbol: symbol.name)

        # check whether the loop was already inserted
        if not any(kernel_loop.variable == loop.variable for kernel_loop in kernel_loops):
         
            # add the loop variable as local variable in the kernel
            call.routine.variables += (loop_variable.clone(scope=call.routine, type=loop_variable_type),)
            if self.dim_name != loop_variable.name:
                call.routine.variables += additional_variables
            # DONE: args vs. kwargs ... just use kwargs!?
            loop_arg_type = loop_variable_type.clone(intent='in')
            # according to "DO loop_variables=loop_start,loop_end,loop_step"
            loop_start = loop_variable.clone(name=f"{loop.variable.name}_start", type=loop_arg_type)
            loop_end = loop_start.clone(name=f"{loop.variable.name}_end")
            loop_step = loop_start.clone(name=f"{loop.variable.name}_step")
            
            # add arguments to the callee
            dimension_size_var = SCCBaseTransformation.get_integer_variable(routine, self.dimension.size)
            call.routine.arguments += (loop_start, loop_end, loop_step, dimension_size_var.clone(type=dimension_size_var.type.clone(intent='in'), scope=call.routine)) # routine.variable_map[self.dimension.size])
            call.routine.arguments += as_tuple(additional_arguments)

            # TODO: remove
            debug_nodes = (ir.Comment(text=f'print *, "{loop.variable.name}_start: ", {loop.variable.name}_start'),
                ir.Comment(text=f'print *, "{loop.variable.name}_end: ", {loop.variable.name}_end'),
                    ir.Comment(text=f'print *, "{loop.variable.name}_step: ", {loop.variable.name}_step'))
            debug_nodes = ()

            call.routine.body = (ir.Comment(text=""), 
                    ir.Comment(text=f"! START of Loki inserted loop {loop.variable}"),
                    loop.clone(bounds=sym.LoopRange((call.routine.variable_map[f"{loop.variable.name}_start"],
                        call.routine.variable_map[f"{loop.variable.name}_end"],
                        call.routine.variable_map[f"{loop.variable.name}_step"])), body=debug_nodes + (dim_name_assignment, call.routine.body,)),
                    ir.Comment(text=f"! END of Loki inserted loop {loop.variable}"))

        start = loop.bounds.children[0]
        end = loop.bounds.children[1]
        step = loop.bounds.children[2] if len(loop.bounds.children) > 2 else sym.IntLiteral(1) # explicitly set step to 1 if it's implicitly 1
        if step is None:
            step = sym.IntLiteral(1)
        additional_kwarguments = [(symbol.name, symbol) for symbol in additional_arguments]
        return ((f"{loop.variable.name}_start", start), (f"{loop.variable.name}_end", end),
                (f"{loop.variable.name}_step", step), (self.dimension.size, SCCBaseTransformation.get_integer_variable(routine, self.dimension.size))) + as_tuple(additional_kwarguments)
        # (routine.variable_map[self.dimension.size].name,
        #             routine.variable_map[self.dimension.size].clone(scope=call.routine))) + as_tuple(additional_kwarguments)

    def process_routine(self, routine, targets, role, remove_loop=True, insert_index_instead_of_loop=False):
        # if DRIVER
        #  find loops with relevant loop index
        #   remove loop, but leave some Loki pragma (so the information is not lost for e.g. launch configuration ...) ?!
        #  find calls and arguments with relevant dimension
        #   remove dimension
        #    either here or separate/independent trafo: convert arr(:, :, :) to arr
        #      if this is not possible throw error if necessary (e.g. for CUF/"C" ...)
        
        # remove "vector_section" (which would be removed by SCCAnnotate ...)
        self._remove_vector_sections(routine)

        # find loops with relevant index/loop variable
        loops = FindNodes(ir.Loop).visit(routine.body)
        loops = [loop for loop in loops if loop.variable == self.dimension.index or loop.variable in self.dimension._aliases]
        
        loop_map = {}
        comment = ir.Comment(text="")
        # find relevant calls within those loops
        relevant_callees = []
        # insert loop in relevant kernel, 
        #  update call arguments and collect information in order to 
        #  promote/update callee arguments/variables in the next step
        if not loops:
            calls = FindNodes(ir.CallStatement).visit(routine.body)
            for call in calls:
                if call.routine.name.lower() not in targets:
                    continue
                if SCCBaseTransformation.is_elemental(call.routine):
                    continue
                if call.routine.name not in self.call_routines:
                    continue
                additional_kwarguments = self._insert_index_in_kernel(routine, call) #  loop)
                call._update(kwarguments=call.kwarguments + additional_kwarguments)
        call_routine_map = {}
        for loop in loops:
            # driver loop mapping
            if remove_loop:
                loop_node = loop if self.keep_driver_loop else loop.body
                loop_map[loop] = (comment, ir.Pragma(keyword="loki", content=f"start: removed loop"), #  l-{loop.bounds.lower}-l u-{loop.bounds.upper}-u s-{loop.bounds.step}-s"),
                        comment, loop_node, comment, ir.Pragma(keyword="loki", content="end: removed loop"), comment)
            # calls
            calls = FindNodes(ir.CallStatement).visit(loop.body)
            # call_routine_map = {}
            for call in calls:
                # only those callees being in targets
                if call.routine.name.lower() not in targets:
                    continue
                
                if SCCBaseTransformation.is_elemental(call.routine):
                    continue

                if insert_index_instead_of_loop:
                    additional_kwarguments = self._insert_index_in_kernel(routine, call, loop) 
                else:
                    additional_kwarguments = self._insert_loop_in_kernel(routine, call, loop)
               
                self.call_routines.append(call.routine.name)
                relevant_callees.append(call.routine)
                call_routine_map[call.routine.name] = {}
                call_routine_map[call.routine.name]['shape'] = {}
                call_routine_map[call.routine.name]['dim'] = {}
                
                new_arguments = ()
                new_kwargs = ()
               
                if self.driver_dim_name is not None: # ignore_dim_name:
                    if role == 'driver':
                        dim_name = self.driver_dim_name
                    else:
                        dim_name = self.dim_name
                else:
                    dim_name = self.dim_name
                # dim_name = self.dim_name
                args_and_kwargs = list(chain(((None, arg) for arg in call.arguments), call.kwarguments)) # caller side
                arg_iter = list(call.arg_iter()) # callee and caller
                for i_arg, (keyword, arg) in enumerate(args_and_kwargs):
                    if isinstance(arg, sym.Array):
                        # replace_indices = [dim == self.dim_name for dim in arg.dimensions]
                        replace_indices = [dim == dim_name for dim in arg.dimensions]
                        corresponding_var = arg_iter[i_arg][0] # call.routine.arguments[i_arg]
                        if any(replace_indices):
                            relevant_index = [i for i, x in enumerate(replace_indices) if x][0]
                            arg_shape = arg.shape
                            call_arg_shape = list(call.routine.variable_map[corresponding_var.name].shape)
                            call_arg_dim = list(call.routine.variable_map[corresponding_var.name].dimensions)
                            call_routine_map[call.routine.name]['shape'][call.routine.variable_map[corresponding_var.name]] = (relevant_index, arg_shape[relevant_index])
                            if loop.variable.name != dim_name: # self.dim_name:
                                call_routine_map[call.routine.name]['dim'][corresponding_var.name] = (relevant_index, self.dim_name)
                            else:
                                call_routine_map[call.routine.name]['dim'][corresponding_var.name] = (relevant_index, loop.variable.name)
                        if arg.dimensions:
                            # new_dimensions = [dim if dim != self.dim_name else sym.RangeIndex((None, None)) for dim in arg.dimensions]
                            new_dimensions = [dim if dim != dim_name else sym.RangeIndex((None, None)) for dim in arg.dimensions]
                        else:
                            # TODO: this should never be called ...
                            new_dimensions = ()
                        # update dimensions of (kw)args
                        if keyword is None:
                            new_arguments += (arg.clone(dimensions=as_tuple(new_dimensions)),)
                        else:
                            new_kwargs += ((keyword, arg.clone(dimensions=as_tuple(new_dimensions))),)
                    else:
                        # nothing to replace, thus, just append as (kw)arg is
                        if keyword is None:
                            new_arguments += (arg,)
                        else:
                            new_kwargs += ((keyword, arg),)
                           
                call._update(arguments=new_arguments, kwarguments=new_kwargs+additional_kwarguments)
        routine.body = Transformer(loop_map).visit(routine.body)

        # promote variables/arrays ...
        relevant_callees = set(relevant_callees)
        for relevant_callee in relevant_callees:

            # promote locals/temporaries (TODO: if set ...)
            # self._promote_locals(relevant_callee)

            # Declarations
            shape_map = {}
            for key, elem in call_routine_map[relevant_callee.name]['shape'].items():
                var_dims = list(key.dimensions)
                var_dims.insert(elem[0], elem[1])
                var_shape = list(key.shape)
                var_shape.insert(elem[0], elem[1])
                shape_map[key] = key.clone(type=key.type.clone(shape=as_tuple(var_shape)), dimensions=as_tuple(var_dims))
            relevant_callee.spec = SubstituteExpressions(shape_map).visit(relevant_callee.spec)
            # Variable usages
            dim_map = {}
            relevant_variables = [key for key in call_routine_map[relevant_callee.name]['dim']]
            variables = FindVariables().visit(relevant_callee.body)
            for var in variables:
                if var.name in relevant_variables:
                    if not var.dimensions:
                        # TODO: why the ... is this necessary?!
                        var_dims = [sym.RangeIndex((None, None))] * (len(var.shape) - 1)
                    else:
                        var_dims = list(var.dimensions)
                   
                    var_dims.insert(call_routine_map[relevant_callee.name]['dim'][var.name][0],
                            relevant_callee.variable_map[call_routine_map[relevant_callee.name]['dim'][var.name][1]])
                    dim_map[var] = var.clone(dimensions=as_tuple(var_dims))
            relevant_callee.body = SubstituteExpressions(dim_map).visit(relevant_callee.body)

