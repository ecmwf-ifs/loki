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
from loki.expression import FindVariables, SubstituteExpressions
from loki.ir import CallStatement, Allocation, Deallocation
from loki.tools.util import is_iterable, as_tuple, CaseInsensitiveDict
from loki.visitors import Transformer, FindNodes
from loki.transform.transformation import Transformation
from loki.transform.transform_utilities import single_variable_declaration
from loki.bulk.item import ProcedureItem
import loki.expression.symbols as sym


__all__ = ['HoistVariablesAnalysis', 'HoistVariablesTransformation',
           'HoistTemporaryArraysAnalysis', 'HoistTemporaryArraysTransformationAllocatable']


class HoistVariablesAnalysis(Transformation):
    """
    **Base class** for the *Analysis* part of the hoist variables functionality/transformation.

    Traverses all subroutines to find the variables to be hoisted.
    Create a derived class and override :func:`find_variables<HoistVariablesAnalysis.find_variables>`
    to define which variables to be hoisted.

    Parameters
    ----------
    key : str
        Access identifier/key for the ``item.trafo_data`` dictionary. Only necessary to provide if several of
        these transformations are carried out in succession.
    """

    _key = 'HoistVariablesTransformation'

    # Apply in reverse order to recursively find all variables to be hoisted.
    reverse_traversal = True

    process_ignored_items = True

    def __init__(self, key=None):
        if key is not None:
            self._key = key

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
            item.trafo_data[self._key]["hoist_variables"] = [var.clone(name=f'{routine.name}_{var.name}')
                                                             for var in variables]
        else:
            item.trafo_data[self._key]["to_hoist"] = []
            item.trafo_data[self._key]["hoist_variables"] = []

        calls = FindNodes(CallStatement).visit(routine.body)
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
    key : str
        Access identifier/key for the ``item.trafo_data`` dictionary. Only necessary to provide if several of
        these transformations are carried out in succession.
    as_kwarguments : boolean
        Whether to pass the hoisted arguments as `args` or `kwargs`.
    """

    _key = 'HoistVariablesTransformation'

    def __init__(self, key=None, as_kwarguments=False):
        if key is not None:
            self._key = key
        self.as_kwarguments = as_kwarguments

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
            self.driver_variable_declaration(routine, item.trafo_data[self._key]["to_hoist"])
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
        for call in FindNodes(CallStatement).visit(routine.body):
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
                call_map[call] = self.kernel_call_argument_remapping(
                    routine=routine, call=call, variables=hoisted_variables
                )

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
            return call.clone(kwarguments=call.kwarguments + new_kwargs)
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
            return call.clone(kwarguments=call.kwarguments + new_kwargs)
        new_args = tuple(v.clone(dimensions=None) for v in variables)
        return call.clone(arguments=call.arguments + new_args)


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
    key : str, optional
        Access identifier/key for the ``item.trafo_data`` dictionary. Only necessary to provide if several of
        these transformations are carried out in succession.
    dim_vars: tuple of str, optional
        Variables to be within the dimensions of the arrays to be hoisted. If not provided, no checks will be done
        for the array dimensions.
    """

    # Apply in reverse order to recursively find all variables to be hoisted.
    reverse_traversal = True

    def __init__(self, key=None, dim_vars=None, **kwargs):
        super().__init__(key=key, **kwargs)
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
        return [var for var in routine.variables
                if var not in routine.arguments    # local variable
                and not var.type.parameter         # not a parameter
                and isinstance(var, sym.Array)     # is an array
                and (self.dim_vars is None         # if dim_vars not empty check if at least one dim is within dim_vars
                     or any(dim_var in self.dim_vars for dim_var in FindVariables().visit(var.dimensions)))]


class HoistTemporaryArraysTransformationAllocatable(HoistVariablesTransformation):
    """
    **Specialisation** for the *Synthesis* part of the hoist variables
    functionality/transformation, to hoist temporary arrays and make
    them ``allocatable``, including the actual *allocation* and
    *de-allocation*.

    Parameters
    ----------
    key : str, optional
        Access identifier/key for the ``item.trafo_data`` dictionary. Only necessary to provide if several of
        these transformations are carried out in succession.
    """

    def __init__(self, key=None, **kwargs):
        super().__init__(key=key, **kwargs)

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
