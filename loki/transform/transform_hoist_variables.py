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
    scheduler.process(transformation=HoistTemporaryArraysAnalysis(), reverse=True)
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
        scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('b',), key=key), reverse=True)
        scheduler.process(transformation=HoistTemporaryArraysTransformation(key=key))
        key = "AnotherUniqueKey"
        scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('a',), key=key), reverse=True)
        scheduler.process(transformation=HoistTemporaryArraysTransformationAllocatable(key=key))
"""
from loki.expression import FindVariables, SubstituteExpressions
from loki.ir import CallStatement, Allocation, Deallocation
from loki.tools.util import is_iterable, as_tuple
from loki.visitors import Transformer, FindNodes
from loki.transform.transformation import Transformation
import loki.expression.symbols as sym


__all__ = ['HoistVariablesAnalysis', 'HoistVariablesTransformation',
           'HoistTemporaryArraysAnalysis', 'HoistTemporaryArraysTransformationAllocatable']


class HoistVariablesAnalysis(Transformation):
    """
    **Base class** for the *Analysis* part of the hoist variables functionality/transformation.

    Traverses all subroutines to find the variables to be hoisted.
    Create a derived class and override :func:`find_variables<HoistVariablesAnalysis.find_variables>`
    to define which variables to be hoisted.

    .. note::
        To be applied **reversed**, in order to recursively find all variables to be hoisted.

    Parameters
    ----------
    key : str
        Access identifier/key for the ``item.trafo_data`` dictionary. Only necessary to provide if several of
        these transformations are carried out in succession.
    """

    _key = 'HoistVariablesTransformation'

    def __init__(self, key=None, disable=None):
        if key is not None:
            self._key = key
        if disable is None:
            self.disable = ()
        else:
            self.disable = [_.upper() for _ in disable]

    def transform_subroutine(self, routine, **kwargs):
        """
        Analysis applied to :any:`Subroutine` item.

        Collects all the variables to be hoisted, including renaming in order to grant for unique variable names.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

        role = kwargs.get('role', None)
        item = kwargs.get('item', None)
        _successors = kwargs.get('successors', ())
        successors = [_ for _ in _successors if _.local_name.upper() not in self.disable]

        if item and not item.local_name == routine.name.lower() or item.local_name.upper() in self.disable:
            return

        item.trafo_data[self._key] = {}

        if role != 'driver':
            variables = self.find_variables(routine)
            item.trafo_data[self._key]["to_hoist"] = variables
            item.trafo_data[self._key]["hoist_variables"] = [var.clone(name=f'{routine.name}_{var.name}')
                                                             for var in variables]
        else:
            item.trafo_data[self._key]["to_hoist"] = []
            item.trafo_data[self._key]["hoist_variables"] = []

        calls = [call for call in FindNodes(CallStatement).visit(routine.body) if call.name
                 not in self.disable]
        call_map = {str(call.name): call for call in calls}

        for child in successors:
            arg_map = dict(call_map[child.routine.name].arg_iter())
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
    """

    _key = 'HoistVariablesTransformation'

    def __init__(self, key=None, disable=None):
        if key is not None:
            self._key = key
        if disable is None:
            self.disable = ()
        else:
            self.disable = [_.upper() for _ in disable]

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
        _successors = kwargs.get('successors', ())
        successors = [_ for _ in _successors if _.local_name.upper() not in self.disable]
        successor_map = {successor.routine.name: successor for successor in successors}

        if item and not item.local_name == routine.name.lower() or item.local_name.upper() in self.disable:
            return

        if self._key not in item.trafo_data:
            raise RuntimeError(f'{self.__class__.__name__} requires key "{self._key}" in item.trafo_data!\n'
                               f'Make sure to call HoistVariablesAnalysis (or any derived class) before and to provide '
                               f'the correct key.')

        if role == 'driver':
            for var in item.trafo_data[self._key]["to_hoist"]:
                self.driver_variable_declaration(routine, var)
        else:
            routine.arguments += as_tuple([var.clone(type=var.type.clone(intent='inout'),
                                                     scope=routine) for var in item.trafo_data[self._key]["to_hoist"]])

        call_map = {}
        calls = [_ for _ in FindNodes(CallStatement).visit(routine.body) if _.name not in self.disable]
        for call in calls:
            new_args = [arg.clone(dimensions=None) for arg
                        in successor_map[str(call.routine.name)].trafo_data[self._key]["hoist_variables"]]
            arguments = list(call.arguments) + new_args
            call_map[call] = call.clone(arguments=as_tuple(arguments))

        routine.body = Transformer(call_map).visit(routine.body)

    def driver_variable_declaration(self, routine, var):
        """
        **Override**: Define the variable declaration (and possibly allocation, de-allocation, ...)
        for each variable to be hoisted.

        Declares hoisted variables with a re-scope.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        var : :any:`Variable`
            The variable to be declared.
        """
        routine.variables += tuple([var.rescope(routine)])


class HoistTemporaryArraysAnalysis(HoistVariablesAnalysis):
    """
    **Specialisation** for the *Analysis* part of the hoist variables functionality/transformation, to hoist only
    temporary arrays and if provided only temporary arrays with specific variables/variable names within the
    array dimensions.

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

    def __init__(self, key=None, disable=None, dim_vars=None, **kwargs):
        super().__init__(key=key, disable=disable, **kwargs)
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
    **Specialisation** for the *Synthesis* part of the hoist variables functionality/transformation, to hoist temporary
    arrays and make them ``allocatable``, including the actual *allocation* and *de-allocation*.

    Parameters
    ----------
    key : str, optional
        Access identifier/key for the ``item.trafo_data`` dictionary. Only necessary to provide if several of
        these transformations are carried out in succession.
    """

    def __init__(self, key=None, disable=None, **kwargs):
        super().__init__(key=key, disable=disable, **kwargs)

    def driver_variable_declaration(self, routine, var):
        """
        Declares hoisted arrays as ``allocatable``, including *allocation* and *de-allocation*.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        var : :any:`Variable`
            The array to be declared, allocated and de-allocated.
        """
        routine.variables += tuple([var.clone(scope=routine, dimensions=as_tuple(
            [sym.RangeIndex((None, None))] * (len(var.dimensions))), type=var.type.clone(allocatable=True))])
        routine.body.prepend(Allocation((var.clone(),)))
        routine.body.append(Deallocation((var.clone(dimensions=None),)))
