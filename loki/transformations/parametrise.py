# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Parametrise variables.

E.g., parametrise

.. code-block:: fortran

    subroutine driver(a, b)
        integer, intent(in) :: a
        integer, intent(in) :: b
        call kernel(a, b)
    end subroutine driver

    subroutine kernel(a, b)
        integer, intent(in) :: a
        integer, intent(in) :: b
        real :: array(a)
        ...
    end subroutine kernel

using the transformation

.. code-block:: python

    dic2p = {'a': 10}
    scheduler.process(transformation=ParametriseTransformation(dic2p=dic2p))

to

.. code-block:: fortran

    subroutine driver(parametrised_a, b)
        integer, parameter :: a = 10
        integer, intent(in) :: parametrised_a
        integer, intent(in) :: b
        IF (parametrised_a /= 10) THEN
            PRINT *, "Variable a parametrised to value 10, but subroutine driver received another value."
            STOP 1
        END IF
        call kernel(b)
    end subroutine driver

    subroutine kernel(b)
        integer, parameter :: a = 10
        integer, intent(in) :: b
        real :: array(a)
        ...
    end subroutine kernel

or

.. code-block:: fortran

    subroutine driver(parametrised_a, b)
        integer, intent(in) :: parametrised_a
        integer, intent(in) :: b
        IF (parametrised_a /= 10) THEN
            PRINT *, "Variable a parametrised to value 10, but subroutine driver received another value."
            STOP 1
        END IF
        call kernel(b)
    end subroutine driver

    subroutine kernel(b)
        integer, intent(in) :: b
        real :: array(10)
        ...
    end subroutine kernel

using the transformation

.. code-block:: python

    dic2p = {'a': 10}
    scheduler.process(transformation=ParametriseTransformation(dic2p=dic2p, replace_by_value=True))
"""

from loki.batch import Transformation
from loki.expression import symbols as sym, LokiIdentityMapper
from loki.ir import nodes as ir, Transformer, FindNodes, FindInlineCalls, SubstituteExpressions, SubstituteStringExpressions
from loki.tools.util import as_tuple, CaseInsensitiveDict

from loki.transformations.utilities import single_variable_declaration
from loki.transformations.inline import inline_constant_parameters
from loki.types import SymbolAttributes, BasicType, DerivedType

__all__ = ['ParametriseTransformation', 'declare_fixed_value_scalars_as_constants', 'parametrise_routine']


# class DerivedTypeMapper(SubstituteExpressionsMapper):
# 
#     def map_derived_type_symbol(self, expr, *args, **kwargs):
#         print(f"map_derived_type_symbol called for expr: {expr}")
#         self.map_variable_symbol(expr, *args, **kwargs)
# 
# class SubstituteDerivedTypeExpressions(SubstituteExpressions):
# 
#     # expr_mapper = DerivedTypeMapper() # LokiIdentityMapper()
#     def __init__(self, expr_map, invalidate_source=True, **kwargs):
#         super().__init__(invalidate_source=invalidate_source, expr_map=expr_map, **kwargs)
# 
#         # Override the static default with a substitution mapper from ``expr_map``
#         self.expr_mapper = DerivedTypeMapper(expr_map) # SubstituteExpressionsMapper(expr_map)
# 
#     def __call__(self, expr, *args, **kwargs):
#         print(f"__call__: {expr}")
#         if expr is None:
#             return None
#         kwargs.setdefault('recurse_to_declaration_attributes', False)
#         return super().__call__(expr, *args, **kwargs)


class SubstituteDimsStringExpressions(SubstituteStringExpressions):

    def visit_VariableDeclaration(self, o, **kwargs):
        """
        For :any:`VariableDeclaration`  or :any:`ProcedureDeclaration`
        we set ``recurse_to_declaration_attributes=True`` to make sure
        properties in the symbol table are updated during dispatch to
        the expression mapper.

        If source invalidation is being requested, we also check the
        associated type (on first symbol) to track changes there.
        """
        kwargs['recurse_to_declaration_attributes'] = False

        # # Store a copy of the old type, as it will be in-place updated
        # old_type = o.symbols[0].type.clone() if self.invalidate_source else None
        # new = super().visit_Node(o, **kwargs)

        # # Check the type if we're tracking source invalidation
        # if self.invalidate_source and o.source:
        #     if old_type != o.symbols[0].type:
        #         new.source.invalidate()
        old_symbols = o.symbols
        new = super().visit_Node(o, **kwargs)
        new_symbols = new.symbols
        # keep old name, take e.g. new dimensions
        # adapted_symbols = tuple(_new.clone(name=_old.name) for _new, _old in zip(new_symbols, old_symbols))
        adapted_symbols = tuple(_old.clone(dimensions=_new.dimensions) if hasattr(_old, 'dimensions') else _old for _new, _old in zip(new_symbols, old_symbols))
        # print(f"old symbols:     {old_symbols}")
        # print(f"new symbols:     {new_symbols}")
        # print(f"adapted_symbols: {adapted_symbols}")
        return o.clone(symbols=adapted_symbols)

    # def visit_Node(self, o, **kwargs):
    #     """
    #     Visit all children of a :any:`Node`.
    #     """
    #     return self.visit(o.children, **kwargs)

    # def visit_Node(self, o, **kwargs):
    #     """
    #     Handler for :any:`Node` objects.

    #     It replaces :data:`o` by :data:`mapper[o]`, if it is in the mapper,
    #     otherwise visits all children before rebuilding the node.
    #     """
    #     print(f"o.children: {o.children}")
    #     # print(f"visit_Node: {o}")
    #     if o in self.mapper:
    #         handle = self.mapper[o]
    #         if handle is None:
    #             # None -> drop /o/
    #             return None

    #         # For one-to-many mappings making sure this is not replaced again
    #         # as it has been inserted by visit_tuple already
    #         if not is_iterable(handle) or o not in handle:
    #             return handle._rebuild(**handle.args)

    #     rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
    #     print(f"rebuilt: {rebuilt}")
    #     return self._rebuild(o, rebuilt)

    # def visit_VariableDeclaration(self, o, **kwargs):
    #     """
    #     For :any:`VariableDeclaration`  or :any:`ProcedureDeclaration`
    #     we set ``recurse_to_declaration_attributes=True`` to make sure
    #     properties in the symbol table are updated during dispatch to
    #     the expression mapper.

    #     If source invalidation is being requested, we also check the
    #     associated type (on first symbol) to track changes there.
    #     """
    #     # # got symbols and dimensions
    #     # new = self.visit_Node(o, **kwargs)
    #     # # new = super().visit_Node(o, **kwargs)
    #     # assert False
    #     # # return o

    #     kwargs['recurse_to_declaration_attributes'] = False

    #     # Store a copy of the old type, as it will be in-place updated
    #     old_type = o.symbols[0].type.clone() if self.invalidate_source else None
    #     # new = super().visit_Node(o, **kwargs)
    #     # new = self.visit_Node(o, **kwargs)
    #     rebuilt = (o.symbols,) + tuple(self.visit(i, **kwargs) for i in (o.dimensions,))
    #     print(f"rebuilt: {rebuilt}")
    #     new = self._rebuild(o, rebuilt)

    #     # Check the type if we're tracking source invalidation
    #     if self.invalidate_source and o.source:
    #         if old_type != o.symbols[0].type:
    #             new.source.invalidate()

    #     return new

def parametrise_routine(routine, dic2p):
    """
    dic2p = {'a': 12, 'b': 11}
    """

    # if 'cloudsc' in routine.name.lower():
    #     breakpoint()

    dic2p = CaseInsensitiveDict(dic2p)
    vars2p = [_.lower() for _ in list(dic2p)]
    # proceed if dictionary with mapping of variables to parametrised is not empty
    # if dic2p:
    #     arguments = []
    #     for arg in routine.arguments:
    #         if arg.name.lower() not in vars2p:
    #             arguments.append(arg)
    #         else:
    #             arguments.append(arg.clone(name=f'parametrised_{arg.name}'))
    #     routine.arguments = arguments

    # remove declarations
    declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    parameter_declarations = []
    # decl_map = {}
    # for decl in declarations:
    #     symbols = []
    #     for smbl in decl.symbols:
    #         if smbl in vars2p:
    #             parameter_declarations.append(decl.clone(symbols=(smbl.clone(
    #                 type=decl.symbols[0].type.clone(parameter=True, intent=None,
    #                                                 initial=sym.IntLiteral(
    #                                                     dic2p[smbl.name]))),))) # or smbl.name?
    #         else:
    #             symbols.append(smbl.clone())

    #         if symbols:
    #             decl_map[decl] = decl.clone(symbols=as_tuple(symbols))
    #         else:
    #             decl_map[decl] = None
    # routine.spec = Transformer(decl_map).visit(routine.spec)

    # # introduce parameter declarations
    # declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    # for parameter_declaration in parameter_declarations:
    #     routine.spec.insert(routine.spec.body.index(declarations[0]), parameter_declaration)

    # stack_type = SymbolAttributes(dtype=BasicType.INTEGER, intent='inout', kind=self.stack_int_type_kind)
    # var_name = f'{self.stack_argument_name}_{self.stack_ptr_name}'
    # stack_arg = Variable(name=var_name, type=stack_type, scope=routine)

    var_type = SymbolAttributes(dtype=BasicType.INTEGER)
    ##
    variable_map = routine.variable_map
    der_type_vars2p = [var.lower() for var in vars2p if "%" in var]
    der_type_var_map = {}
    print()
    for der_type_var in vars2p: # der_type_vars2p:
        parent_der_type_var = der_type_var.split('%', maxsplit=1)[0]
        if parent_der_type_var in variable_map:
            der_type_var_map[der_type_var] = f'loki_{der_type_var.replace("%", "_")}'
            parameter_declarations.append(ir.VariableDeclaration(symbols=(sym.Variable(name=f'loki_{der_type_var.replace("%", "_")}', type=var_type.clone(parameter=True, initial=sym.IntLiteral(dic2p[der_type_var]))),)))

    # introduce parameter declarations
    declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    for parameter_declaration in parameter_declarations:
        routine.spec.insert(routine.spec.body.index(declarations[0]), parameter_declaration)

    # routine.body = SubstituteDerivedTypeExpressions({}).visit(routine.body)
    routine.spec = SubstituteDimsStringExpressions(der_type_var_map, scope=routine).visit(routine.spec)
    routine.body = SubstituteStringExpressions(der_type_var_map, scope=routine).visit(routine.body)

def parametrise_routine_backup(routine, dic2p):
    """
    dic2p = {'a': 12, 'b': 11}
    """

    # if 'cloudsc' in routine.name.lower():
    #     breakpoint()

    dic2p = CaseInsensitiveDict(dic2p)
    vars2p = [_.lower() for _ in list(dic2p)]
    # proceed if dictionary with mapping of variables to parametrised is not empty
    if dic2p:
        arguments = []
        for arg in routine.arguments:
            if arg.name.lower() not in vars2p:
                arguments.append(arg)
            else:
                arguments.append(arg.clone(name=f'parametrised_{arg.name}'))
        routine.arguments = arguments

    # remove declarations
    declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    parameter_declarations = []
    decl_map = {}
    for decl in declarations:
        symbols = []
        for smbl in decl.symbols:
            if smbl in vars2p:
                parameter_declarations.append(decl.clone(symbols=(smbl.clone(
                    type=decl.symbols[0].type.clone(parameter=True, intent=None,
                                                    initial=sym.IntLiteral(
                                                        dic2p[smbl.name]))),))) # or smbl.name?
            else:
                symbols.append(smbl.clone())

            if symbols:
                decl_map[decl] = decl.clone(symbols=as_tuple(symbols))
            else:
                decl_map[decl] = None
    routine.spec = Transformer(decl_map).visit(routine.spec)

    # # introduce parameter declarations
    # declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    # for parameter_declaration in parameter_declarations:
    #     routine.spec.insert(routine.spec.body.index(declarations[0]), parameter_declaration)

    # stack_type = SymbolAttributes(dtype=BasicType.INTEGER, intent='inout', kind=self.stack_int_type_kind)
    # var_name = f'{self.stack_argument_name}_{self.stack_ptr_name}'
    # stack_arg = Variable(name=var_name, type=stack_type, scope=routine)

    var_type = SymbolAttributes(dtype=BasicType.INTEGER)
    ##
    variable_map = routine.variable_map
    der_type_vars2p = [var.lower() for var in vars2p if "%" in var]
    der_type_var_map = {}
    print()
    for der_type_var in der_type_vars2p:
        parent_der_type_var = der_type_var.split('%', maxsplit=1)[0]
        if parent_der_type_var in variable_map:
            der_type_var_map[der_type_var] = f'loki_{der_type_var.replace("%", "_")}'
            parameter_declarations.append(ir.VariableDeclaration(symbols=(sym.Variable(name=f'loki_{der_type_var.replace("%", "_")}', type=var_type.clone(parameter=True, initial=sym.IntLiteral(dic2p[der_type_var]))),)))

    # introduce parameter declarations
    declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    for parameter_declaration in parameter_declarations:
        routine.spec.insert(routine.spec.body.index(declarations[0]), parameter_declaration)

    # routine.body = SubstituteDerivedTypeExpressions({}).visit(routine.body)
    routine.body = SubstituteStringExpressions(der_type_var_map, scope=routine).visit(routine.body)


def declare_fixed_value_scalars_as_constants(routine):
    """
    Mark local scalars that are assigned a fixed value as parameters.

    This is not really sophisticated and will eventually be superseded by
    a constant propagation transformation.
    """
    def is_constant_rhs(expr):
        # expr is a literal e.g., a IntLiteral
        if issubclass(type(expr), sym._Literal):
            return True
        # expr is a Product/Sum and all children are literals
        if isinstance(expr, (sym.Product, sym.Sum)):
            return all(issubclass(type(_expr), sym._Literal) for _expr in expr.children)
        return False

    assignments = FindNodes(ir.Assignment).visit(routine.body)
    # filter for local variables and scalars
    variables = [var for var in routine.variables if var not in routine.arguments and not isinstance(var, sym.Array)]
    # don't bother with those being used in (inline) calls (although intent 'in' would be fine)
    calls = as_tuple(FindNodes(ir.CallStatement).visit(routine.body)) + as_tuple(FindInlineCalls().visit(routine.body))
    args = set()
    for call in calls:
        args |= set(call.arguments) | set(arg[1] for arg in call.kwarguments)
    if args:
        variables = [var for var in variables if var not in args]
    assignments_dic = {}
    for assignment in assignments:
        if assignment.lhs in variables:
            assignments_dic.setdefault(assignment.lhs, []).append(assignment)
    # remove those which are written to multiple times
    keys2remove = []
    for key, vals in assignments_dic.items():
        if len(vals) > 1:
            keys2remove.append(key)
    for key in keys2remove:
        del assignments_dic[key]
    # keep only those which are assigned a constant value to
    parametrise_map = {}
    for key, vals in assignments_dic.items():
        val = vals[0]
        if is_constant_rhs(val.rhs):
            parametrise_map[key] = val
    _vars = list(parametrise_map.keys())
    # make sure the relevant variables are declared individually
    single_variable_declaration(routine, [str(var.name) for var in _vars])
    # update relevant vars to be parameters and assign correct initial value
    for var in _vars:
        routine.symbol_attrs[str(var.name)] = var.type.clone(parameter=True, initial=parametrise_map[var].rhs)
    # remove the original assignments in the body which are now used to initialise the parameter variables
    assignment_map = {}
    for assignment in parametrise_map.values():
        assignment_map[assignment] = None
    routine.body = Transformer(assignment_map).visit(routine.body)


class ParametriseTransformation(Transformation):
    """
    Parametrise variables with provided values.

    This transformation checks for each subroutine (defined as driver or entry point) the arguments to be parametrised
    according to :attr:`dic2p` and passes this information down the calltree.

    .. note::

        A sanity run-time check will be inserted at each entry point to check consistency of the provided value
        and argument value at this point!

    .. warning::

        The subroutine/call signature(s) may be altered as arguments are converted to local parameters or int literals.
        Therefore, consistency must be ensured, meaning all parts of the code calling subroutines that are transformed
        and all possibly differing names of variables at the entry points must be included, otherwise the resulting
        code will not compile correctly!

    E.g., use this class like this:

    .. code-block:: python

        def error_stop(**kwargs):
            msg = kwargs.get("msg")
            return ir.Intrinsic(text=f'error stop "{msg}"'),

        dic2p = {'a': 12, 'b': 11}

        transformation = ParametriseTransformation(dic2p=dic2p, abort_callback=error_stop,
                                entry_points=("driver1", "driver2"))

        scheduler.process(transformation=transformation)

    Parameters
    ----------
    dic2p: dict
        Dictionary of variable names and corresponding values to be parametrised.
    replace_by_value: bool
        Replace variables entirely by value (default: `False`)
    entry_points: None or tuple
        Subroutine names to be used as entry points for parametrisation. Default `None` uses driver(s) as
        entry points.
    abort_callback:
        Callback routine used for error on sanity check.
        Available arguments via ``kwargs``:

        * ``msg`` - predefined error message
        * ``routine`` - the routine executing the sanity check
        * ``var`` - the variable getting checked
        * ``value`` - the value the variable should have (according to :attr:`dic2p`)
    key : str
        Access identifier/key for the ``item.trafo_data`` dictionary. Only necessary to provide if several of
        these transformations are carried out in succession.
    """

    _key = "ParametriseTransformation"

    def __init__(self, dic2p, replace_by_value=False, entry_points=None, abort_callback=None, key=None):
        self.dic2p = dic2p
        self.replace_by_value = replace_by_value
        self.entry_points = tuple(entry_point.upper() for entry_point in as_tuple(entry_points)) or None
        self.abort_callback = abort_callback
        if key is not None:
            self._key = key

    def transform_subroutine(self, routine, **kwargs):
        """
        Transformation applied to :any:`Subroutine` item.

        Parametrises all variables as defined by :attr:`dic2p` either to be a parameter or by replacing the
        variable with the value itself.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

        item = kwargs.get('item', None)
        role = kwargs.get('role', None)
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = as_tuple(sub_sgraph.successors(item)) if sub_sgraph is not None else ()

        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor)
            for successor in successors
        )

        # decide whether subroutine is an entry point or not
        process_entry_point = False
        if self.entry_points is None:
            if role is not None and role == "driver":
                dic2p = self.dic2p
                process_entry_point = True
            else:
                if self._key in item.trafo_data:
                    dic2p = item.trafo_data[self._key]
                else:
                    dic2p = {}
        else:
            if routine.name.upper() in self.entry_points:
                dic2p = self.dic2p
                process_entry_point = True
            else:
                if self._key in item.trafo_data:
                    dic2p = item.trafo_data[self._key]
                else:
                    dic2p = {}

        vars2p = list(dic2p)

        # proceed if dictionary with mapping of variables to parametrised is not empty
        if dic2p:
            if process_entry_point:
                # rename arguments that are parametrised (to allow for sanity checks)
                arguments = []
                for arg in routine.arguments:
                    if arg.name not in vars2p:
                        arguments.append(arg)
                    else:
                        arguments.append(arg.clone(name=f'parametrised_{arg.name}'))
                routine.arguments = arguments
                # introduce sanity check
                for key, value in reversed(dic2p.items()):
                    if f'parametrised_{key}' in routine.variable_map:
                        error_msg = f"Variable {key} parametrised to value {value}, but subroutine {routine.name} " \
                                    f"received another value"
                        condition = sym.Comparison(routine.variable_map[f'parametrised_{key}'], '!=',
                                                   sym.IntLiteral(value))
                        comment = ir.Comment(f"! Stop execution: {error_msg}")
                        parametrised_var = routine.variable_map[f'parametrised_{key}']
                        # use default abort mechanism
                        if self.abort_callback is None:
                            abort = (ir.Intrinsic(text=f'PRINT *, "{error_msg}: ", {parametrised_var.name}'),
                                     ir.Intrinsic(text="STOP 1"))
                        # use user define abort/warn mechanism
                        else:
                            kwargs = {"msg": error_msg, "routine": routine.name, "var": parametrised_var,
                                      "value": value}
                            abort = self.abort_callback(**kwargs)
                        body = (comment,) + abort
                        conditional = ir.Conditional(condition=condition,
                                                     body=body, else_body=None)
                        routine.body.prepend(conditional)
                        routine.body.prepend(ir.Comment(f"! Sanity check for parametrised variable: {key}"))
            else:
                routine.arguments = [arg for arg in routine.arguments if arg.name not in vars2p]

            # remove variables to be parametrised from all call statements
            call_map = {}
            for call in FindNodes(ir.CallStatement).visit(routine.body):
                if str(call.name) in successor_map:
                    successor_map[str(call.name)].trafo_data[self._key] = {}
                    arg_map = dict(call.arg_iter())
                    arg_map_reversed = {v: k for k, v in arg_map.items()}
                    indices = [call.arguments.index(var2p) for var2p in vars2p if var2p in call.arguments]
                    for index in indices:
                        name = str(call.name)
                        successor_map[name].trafo_data[self._key][str(arg_map_reversed[call.arguments[index]])] = \
                            dic2p[call.arguments[index].name]
                    arguments = tuple(arg for arg in call.arguments if arg not in vars2p)
                    call_map[call] = call.clone(arguments=arguments)
            routine.body = Transformer(call_map).visit(routine.body)

            # remove declarations
            declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
            parameter_declarations = []
            decl_map = {}
            for decl in declarations:
                symbols = []
                for smbl in decl.symbols:
                    if smbl in vars2p:
                        parameter_declarations.append(decl.clone(symbols=(smbl.clone(
                            type=decl.symbols[0].type.clone(parameter=True, intent=None,
                                                            initial=sym.IntLiteral(
                                                                dic2p[smbl.name]))),))) # or smbl.name?
                    else:
                        symbols.append(smbl.clone())

                    if symbols:
                        decl_map[decl] = decl.clone(symbols=as_tuple(symbols))
                    else:
                        decl_map[decl] = None
            routine.spec = Transformer(decl_map).visit(routine.spec)

            # introduce parameter declarations
            declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
            for parameter_declaration in parameter_declarations:
                routine.spec.insert(routine.spec.body.index(declarations[0]), parameter_declaration)

            # replace all parameter variables with their corresponding value (inline constant parameters)
            if self.replace_by_value:
                inline_constant_parameters(routine=routine, external_only=False)
