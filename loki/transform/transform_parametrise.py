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
from loki.expression import symbols as sym
from loki import ir, FindVariables, SubstituteExpressions
from loki.visitors import Transformer, FindNodes
from loki.tools.util import as_tuple, CaseInsensitiveDict
from loki.transform.transformation import Transformation
from loki.transform.transform_inline import inline_constant_parameters

__all__ = ['ParametriseTransformation', 'ParametriseArrayDimsTransformation']


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

        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor)
            for successor in as_tuple(kwargs.get('successors'))
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

class ParametriseArrayDimsTransformation(Transformation):


    _key = "ParametriseTransformation"
    def __init__(self, dic2p, replace_by_value=False, entry_points=None, abort_callback=None, key=None):
        self.dic2p = dic2p
        if key is not None:
            self._key = key

    def transform_subroutine(self, routine, **kwargs):

        var2p = dict(self.dic2p)
        variables = list(routine.variables)
        variables += list(FindVariables(unique=False).visit(routine.body))
        
        introduce_loki_params = []
        var_map = {}
        vmap_dims = {}
        clone_var = False
        for var in variables:
            if not isinstance(var, sym.Array):
                continue
            # for dim in var.shape:
            #     dim_var_map = {}
            #     dim_vars = FindVariables(unique=False).visit(dim)
            #     for dim_var in dim_vars:
            #         if dim_var.name.lower() in vars2p:
            #             dim_var_map[dim_var] = sym.Variable(name=f"{dim_var.name}_loki_param")
            new_shape = ()
            clone_var = False
            for dim in var.shape:
                if dim in var2p:
                    new_shape += (dim.clone(name=f"{dim.name}_loki_param"),)
                    vmap_dims[dim] = dim.clone(name=f"{dim.name}_loki_param")
                    introduce_loki_params.append(dim)
                    clone_var = True
                else:
                    new_shape += (dim,)
            # if new_shape != var.shape:
            if clone_var:
                var_map[var] = var.clone(type=var.type.clone(shape=new_shape), dimensions=new_shape)

        print(f"herehere var_map: {var_map}")
        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)
       
        for var in list(dict.fromkeys(introduce_loki_params)):
            routine.variables += (var.clone(name=f"{var.name}_loki_param", type=var.type.clone(intent=None, initial=sym.IntLiteral(self.dic2p[var.name.lower()]))),)

        print(f"herehere vmap_dims: {vmap_dims}")
        for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
            if decl.symbols[0].name in [var.name for var in variables]:
                print(f"updating dimensions for decl: {decl}")
                try:
                    print(f" herehere updating to {vmap_dims[decl.symbols[0].name]}")
                    decl._update(dimensions=vmap_dims[decl.symbols[0].name])
                except:
                    decl._update(dimensions=None)

