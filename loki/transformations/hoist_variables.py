# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Multiple **transformations to hoist variables** especially to hoist temporary arrays.

E.g., the following source code

.. code-block:: fortran

    subroutine driver(...)
        integer :: a
        a = 10
        call kernel(a)
    end subroutine driver

    subroutine kernel(a)
        integer, intent(in) :: a
        real :: array(a)
        ...
    end subroutine kernel

can be transformed/hoisted to

.. code-block:: fortran

    subroutine driver(...)
        integer :: a
        real :: kernel_array(a)
        a = 10
        call kernel(a, kernel_array)
    end subroutine driver

    subroutine kernel(a, array)
        integer, intent(in) :: a
        real, intent(inout) :: array(a)
        ...
    end subroutine kernel

using

.. code-block:: python

    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis())
    # Transformation: Synthesis
    scheduler.process(transformation=HoistVariablesTransformation())


To achieve this two transformation are necessary, whereas the first one is responsible for the *Analysis* and the
second one for the *Synthesis*. Two base classes

* :class:`.HoistVariablesAnalysis` - *Analysis* part, to be processed in reverse
    * specialise/implement :func:`find_variables<HoistVariablesAnalysis.find_variables>`
* :class:`.HoistVariablesTransformation`- *Synthesis* part
    * specialise/implement :func:`driver_variable_declaration<HoistVariablesSynthesis.driver_variable_declaration>`

are provided to create derived classes for specialisation of the actual hoisting.

.. warning::
    :class:`.HoistVariablesAnalysis` ensures that all local variables are hoisted!
    Please consider using a specialised class like :class:`.HoistTemporaryArraysAnalysis` or create a derived class
    yourself.

.. note::
    If several of these transformations are carried out in succession, provide a unique ``key`` for each corresponding
    *Analysis* and *Synthesis* step!

    .. code-block:: python

        key = "UniqueKey"
        scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('b',), key=key))
        scheduler.process(transformation=HoistTemporaryArraysTransformation(key=key))
        key = "AnotherUniqueKey"
        scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('a',), key=key))
        scheduler.process(transformation=HoistTemporaryArraysTransformationAllocatable(key=key))
"""

from collections import defaultdict

from loki.batch import Transformation, ProcedureItem
from loki.expression import (
    symbols as sym, FindVariables, FindInlineCalls,
    SubstituteExpressions, is_dimension_constant
)
from loki.ir import (
    CallStatement, Allocation, Deallocation, Transformer, FindNodes, Comment, Import,
    Assignment
)
from loki.tools.util import is_iterable, as_tuple, CaseInsensitiveDict, flatten

from loki.transformations.utilities import single_variable_declaration


__all__ = [
    'HoistVariablesAnalysis', 'HoistVariablesTransformation',
    'HoistTemporaryArraysAnalysis', 'HoistTemporaryArraysTransformationAllocatable'
]


class HoistVariablesAnalysis(Transformation):
    """
    **Base class** for the *Analysis* part of the hoist variables functionality/transformation.

    Traverses all subroutines to find the variables to be hoisted.
    Create a derived class and override :func:`find_variables<HoistVariablesAnalysis.find_variables>`
    to define which variables to be hoisted.
    """

    _key = 'HoistVariablesTransformation'

    # Apply in reverse order to recursively find all variables to be hoisted.
    reverse_traversal = True

    process_ignored_items = True

    def transform_subroutine(self, routine, **kwargs):
        """
        Analysis applied to :any:`Subroutine` item.

        Collects all the variables to be hoisted, including renaming
        in order to grant for unique variable names.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

        role = kwargs.get('role', None)
        item = kwargs.get('item', None)
        successors = as_tuple(kwargs.get('successors'))

        item.trafo_data[self._key] = {}

        if role != 'driver':
            variables = self.find_variables(routine)
            item.trafo_data[self._key]["to_hoist"] = variables
            dims = flatten([getattr(v, 'shape', []) for v in variables])
            import_map = routine.import_map
            item.trafo_data[self._key]["imported_sizes"] = [(d.type.module, d) for d in dims
                                                            if str(d) in import_map]
            item.trafo_data[self._key]["hoist_variables"] = [var.clone(name=f'{routine.name}_{var.name}')
                                                             for var in variables]
        else:
            item.trafo_data[self._key]["imported_sizes"] = []
            item.trafo_data[self._key]["to_hoist"] = []
            item.trafo_data[self._key]["hoist_variables"] = []

        calls = FindNodes(CallStatement).visit(routine.body)
        calls += FindInlineCalls().visit(routine.body)
        call_map = CaseInsensitiveDict((str(call.name), call) for call in calls)

        for child in successors:
            if not isinstance(child, ProcedureItem):
                continue

            arg_map = dict(call_map[child.local_name].arg_iter())
            hoist_variables = []
            for var in child.trafo_data[self._key]["hoist_variables"]:
                if isinstance(var, sym.Array):
                    dimensions = SubstituteExpressions(arg_map).visit(var.dimensions)
                    hoist_variables.append(var.clone(dimensions=dimensions, type=var.type.clone(shape=dimensions)))
                else:
                    hoist_variables.append(var)
            item.trafo_data[self._key]["to_hoist"].extend(hoist_variables)
            item.trafo_data[self._key]["to_hoist"] = list(dict.fromkeys(item.trafo_data[self._key]["to_hoist"]))
            item.trafo_data[self._key]["hoist_variables"].extend(hoist_variables)
            item.trafo_data[self._key]["hoist_variables"] = list(dict.fromkeys(
                item.trafo_data[self._key]["hoist_variables"]))
            item.trafo_data[self._key]["imported_sizes"] += child.trafo_data[self._key]["imported_sizes"]

    def find_variables(self, routine):
        """
        **Override**: Find/Select all the variables to be hoisted.

        Selects all local variables that are not ``parameter`` to be hoisted.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine find the variables.
        """
        return [var for var in routine.variables if var not in routine.arguments if not var.type.parameter]


class HoistVariablesTransformation(Transformation):
    """
    **Base class** for the *Synthesis* part of the hoist variables functionality/transformation.

    Traverses all subroutines to hoist the variables.
    Create a derived class and override :func:`find_variables<HoistVariablesAnalysis.find_variables>`
    to define which variables to be hoisted.

    .. note::
        Needs the *Analysis* part to be processed first in order to hoist all already found variables.

    Parameters
    ----------
    as_kwarguments : boolean
        Whether to pass the hoisted arguments as `args` or `kwargs`.
    remap_dimensions : boolean
        Remap dimensions based on variables that are used for initializing
        other variables that could end up as dimensions for hoisted arrays.
        Thus, account for possibly uninitialized variables used as dimensions.
    """

    _key = 'HoistVariablesTransformation'

    def __init__(self, as_kwarguments=False, remap_dimensions=False):
        self.as_kwarguments = as_kwarguments
        self.remap_dimensions = remap_dimensions

    def transform_subroutine(self, routine, **kwargs):
        """
        Transformation applied to :any:`Subroutine` item.

        Hoists all to be hoisted variables which includes

        * appending the arguments for each subroutine
        * appending the arguments for each subroutine call
        * modifying the variable declaration in the subroutine
        * adding the variable declaration in the driver

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """
        role = kwargs.get('role', None)
        item = kwargs.get('item', None)
        successors = as_tuple(kwargs.get('successors'))
        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor) for successor in successors
        )

        if self._key not in item.trafo_data:
            raise RuntimeError(f'{self.__class__.__name__} requires key "{self._key}" in item.trafo_data!\n'
                               f'Make sure to call HoistVariablesAnalysis (or any derived class) before and to provide '
                               f'the correct key.')

        if role == 'driver':
            if self.remap_dimensions:
                to_hoist = self.driver_variable_declaration_dim_remapping(routine,
                        item.trafo_data[self._key]["to_hoist"])
            else:
                to_hoist = item.trafo_data[self._key]["to_hoist"]
            self.driver_variable_declaration(routine, to_hoist)
        else:
            # We build the list of temporaries that are hoisted to the calling routine
            # Because this requires adding an intent, we need to make sure they are not
            # declared together with non-hoisted variables
            hoisted_temporaries = tuple(
                var.clone(type=var.type.clone(intent='inout'), scope=routine)
                for var in item.trafo_data[self._key]['to_hoist']
            )
            single_variable_declaration(routine, variables=[var.clone(dimensions=None) for var in hoisted_temporaries])
            routine.arguments += hoisted_temporaries

        call_map = {}
        for call in FindNodes(CallStatement).visit(routine.body) + list(FindInlineCalls().visit(routine.body)):
            # Only process calls in this call tree
            if str(call.name) not in successor_map:
                continue

            successor_item = successor_map[str(call.routine.name)]
            if self.as_kwarguments:
                to_hoist = successor_item.trafo_data[self._key]["to_hoist"]
                _hoisted_variables = successor_item.trafo_data[self._key]["hoist_variables"]
                hoisted_variables = zip(to_hoist, _hoisted_variables)
            else:
                hoisted_variables = successor_item.trafo_data[self._key]["hoist_variables"]
            if role == "driver":
                call_map[call] = self.driver_call_argument_remapping(
                    routine=routine, call=call, variables=hoisted_variables
                )
            elif role == "kernel":
                if isinstance(call, CallStatement):
                    call_map[call] = self.kernel_call_argument_remapping(
                        routine=routine, call=call, variables=hoisted_variables
                    )
                else:
                    self.kernel_inline_call_argument_remapping(
                        routine=routine, call=call, variables=hoisted_variables
                    )

        # Add imports used to define hoisted
        missing_imports_map = defaultdict(set)
        import_map = routine.import_map
        for module, var in item.trafo_data[self._key]["imported_sizes"]:
            if not var.name in import_map:
                missing_imports_map[module] |= {var}

        if missing_imports_map:
            routine.spec.prepend(Comment(text=(
                '![Loki::HoistVariablesTransformation] ---------------------------------------'
            )))
            for module, variables in missing_imports_map.items():
                routine.spec.prepend(Import(module=module.name, symbols=variables))

            routine.spec.prepend(Comment(text=(
                '![Loki::HoistVariablesTransformation] '
                '-------- Added hoisted temporary size imports -------------------------------'
            )))

        routine.body = Transformer(call_map).visit(routine.body)

    def driver_variable_declaration(self, routine, variables):
        """
        **Override**: Define the variable declaration (and possibly
        allocation, de-allocation, ...)  for each variable to be
        hoisted.

        Declares hoisted variables with a re-scope.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        variables : tuple of :any:`Variable`
            The tuple of variables to be declared.
        """
        routine.variables += tuple(v.rescope(routine) for v in variables)

    @staticmethod
    def driver_variable_declaration_dim_remapping(routine, variables):
        """
        Take a list of variables and remap their dimensions for those being
        arrays to account for possibly uninitialized variables/dimensions. 

        Parameters
        ----------
        routine : :any:`Subroutine`
            The relevant subroutine.
        variables : tuple of :any:`Variable`
            The tuple of variables for remapping.
        """
        dim_map = {}
        assignments = FindNodes(Assignment).visit(routine.body)
        for assignment in assignments:
            dim_map[assignment.lhs] = assignment.rhs
        variables = [var.clone(dimensions=SubstituteExpressions(dim_map).visit(var.dimensions))
                if isinstance(var, sym.Array) else var for var in variables]
        return variables

    def driver_call_argument_remapping(self, routine, call, variables):
        """
        Callback method to re-map hoisted arguments for the driver-level routine.

        The callback will simply add all the hoisted variable arrays to the call
        without dimension range symbols.

        This callback is used to adjust the argument variable mapping, so that
        the call signature in the driver can be adjusted to the declaration
        scheme of subclassed variants of the basic hoisting tnansformation.
        Potentially, different variants of the hoist transformation can override
        the behaviour here to map to a differnt call invocation scheme.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        call : :any:`CallStatement`
            Call object to which hoisted variables will be added.
        variables : tuple of :any:`Variable`
            The tuple of variables to be declared.
        as_kwarguments : boolean
            Whether to pass the hoisted arguments as `args` or `kwargs`.
        """
        # pylint: disable=unused-argument
        if self.as_kwarguments:
            new_kwargs = tuple((a.name, v.clone(dimensions=None)) for (a, v) in variables)
            kwarguments = call.kwarguments if call.kwarguments is not None else ()
            return call.clone(kwarguments=kwarguments + new_kwargs)
        new_args = tuple(v.clone(dimensions=None) for v in variables)
        return call.clone(arguments=call.arguments + new_args)

    def kernel_call_argument_remapping(self, routine, call, variables):
        """
        Callback method to re-map hoisted arguments in kernel-to-kernel calls.

        The callback will simply add all the hoisted variable arrays to the call
        without dimension range symbols.
        This callback is used to adjust the argument variable mapping, so that
        the call signature can be adjusted to the declaration
        scheme of subclassed variants of the basic hoisting transformation.
        Potentially, different variants of the hoist transformation can override
        the behaviour here to map to a different call invocation scheme.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        call : :any:`CallStatement`
            Call object to which hoisted variables will be added.
        variables : tuple of :any:`Variable`
            The tuple of variables to be declared.
        """
        # pylint: disable=unused-argument
        if self.as_kwarguments:
            new_kwargs = tuple((a.name, v.clone(dimensions=None)) for (a, v) in variables)
            kwarguments = call.kwarguments if call.kwarguments is not None else ()
            return call.clone(kwarguments=kwarguments + new_kwargs)
        new_args = tuple(v.clone(dimensions=None) for v in variables)
        return call.clone(arguments=call.arguments + new_args)

    def kernel_inline_call_argument_remapping(self, routine, call, variables):
        """
        Append hoisted temporaries to inline function call arguments.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        call : :any:`InlineCall`
            ProcedureSymbol to which hoisted variables will be added.
        variables : tuple of :any:`Variable`
            The tuple of variables to be declared.
        """

        if self.as_kwarguments:
            kw_params = call.kw_parameters
            kw_params.update(dict((a.name, v.clone(dimensions=None)) for (a, v) in variables))
            _call_clone = call.clone(kw_parameters=kw_params)
            vmap = {call: _call_clone}
        else:
            new_args = tuple(v.clone(dimensions=None) for v in variables)
            vmap = {call: call.clone(parameters=call.parameters + new_args)}

        routine.body = SubstituteExpressions(vmap).visit(routine.body)

class HoistTemporaryArraysAnalysis(HoistVariablesAnalysis):
    """
    **Specialisation** for the *Analysis* part of the hoist variables
    functionality/transformation, to hoist only temporary arrays and
    if provided only temporary arrays with specific variables/variable
    names within the array dimensions.

    .. code-block::python

        scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('a',)), reverse=True)
        scheduler.process(transformation=HoistVariablesTransformation())

    Parameters
    ----------
    dim_vars: tuple of str, optional
        Variables to be within the dimensions of the arrays to be
        hoisted. If not provided, no checks will be done for the array
        dimensions.
    """

    # Apply in reverse order to recursively find all variables to be hoisted.
    reverse_traversal = True

    def __init__(self, dim_vars=None):
        self.dim_vars = dim_vars
        if self.dim_vars is not None:
            assert is_iterable(self.dim_vars)

    def find_variables(self, routine):
        """
        Selects temporary arrays to be hoisted.

        * if ``dim_vars`` is ``None`` (default) all temporary arrays will be hoisted
        * if ``dim_vars`` is defined, all arrays with the corresponding dimensions will be hoisted

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine find the variables.
        """

        # Determine function result variable name
        if not (result_name := routine.result_name):
            result_name = routine.name

        variables = [var for var in routine.variables if isinstance(var, sym.Array)]
        return [var for var in variables
                if var not in routine.arguments    # local variable
                and not all(is_dimension_constant(d) for d in var.shape)
                and not var.name.lower() == result_name.lower()
                and (self.dim_vars is None         # if dim_vars not empty check if at least one dim is within dim_vars
                     or any(dim_var in self.dim_vars for dim_var in FindVariables().visit(var.dimensions)))]


class HoistTemporaryArraysTransformationAllocatable(HoistVariablesTransformation):
    """
    **Specialisation** for the *Synthesis* part of the hoist variables
    functionality/transformation, to hoist temporary arrays and make
    them ``allocatable``, including the actual *allocation* and
    *de-allocation*.
    """

    def driver_variable_declaration(self, routine, variables):
        """
        Declares hoisted arrays as ``allocatable``, including *allocation* and *de-allocation*.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        variables : tuple of :any:`Variable`
            The array to be declared, allocated and de-allocated.
        """
        for var in variables:
            routine.variables += as_tuple(
                var.clone(
                    dimensions=as_tuple([sym.RangeIndex((None, None))] * len(var.dimensions)),
                    type=var.type.clone(allocatable=True), scope=routine
                )
            )
            routine.body.prepend(Allocation((var.clone(),)))
            routine.body.append(Deallocation((var.clone(dimensions=None),)))
