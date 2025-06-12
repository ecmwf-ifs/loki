# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Base class definition for :ref:`transformations`.
"""
from pprint import pformat

from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.batch.item import ProcedureItem, ModuleItem


__all__ = ['Transformation', 'TransformationError']


class TransformationError(Exception):
    """
    Exception raised when a :any:`Transformation` encounters an
    error while processing an IR node

    Parameters
    ----------
    message : str
        Description of the error
    transformation : subclass of :any:`Transformation`
        The class of the transformation in which the error occured
    source : :any:`Sourcefile` or :any:`ProgramUnit`
        The object that was processed when the error occured
    """

    def __init__(self, message, transformation, source):
        self.message = message
        self.transformation = transformation
        self.source = source

    def __str__(self):
        return f"Applying {self.transformation.__name__} to {self.source} failed: {self.message}"


class Transformation:
    """
    Base class for source code transformations that manipulate source
    items like :any:`Subroutine` or :any:`Module` in place via the
    ``item.apply(transform)`` method.

    The source transformations to be applied should be defined in the
    following class-specific methods:

    * :meth:`transform_subroutine`
    * :meth:`transform_module`
    * :meth:`transform_file`

    The generic dispatch mechanism behind the :meth:`apply` method will ensure
    that all hierarchies of the data model are traversed and the specific
    method for each level is applied, if the relevant recursion mode is enabled
    in the transformation's manifest (:attr:`recurse_to_modules` and/or
    :attr:`recurse_to_procedures`). Note that in :any:`Sourcefile` objects,
    :any:`Module` members will be traversed before standalone :any:`Subroutine` objects.

    Classes inheriting from :any:`Transformation` may configure the
    invocation and behaviour during batch processing via a predefined
    set of class attributes. These flags determine the underlying
    graph traversal when processing complex call trees and determine
    how the transformations are invoked for a given type of :any:`Item` in the
    :any:`Scheduler`.

    Attributes
    ----------
    reverse_traversal : bool
        Forces scheduler traversal in reverse order from the leaf
        nodes upwards (default: ``False``).
    traverse_file_graph : bool
         Apply :any:`Transformation` to the :any:`Sourcefile` object
         corresponding to the :any:`Item` being processed, instead of
         the program unit in question (default: ``False``).
    item_filter : bool
        Filter by graph node types to prune the graph and change connectivity.
        By default, only calls to :any:`Subroutine` items are used to construct
        the graph.
    recurse_to_modules : bool
        Apply transformation to all :any:`Module` objects when processing
        a :any:`Sourcefile` (default ``False``)
    recurse_to_procedures : bool
        Apply transformation to all :any:`Subroutine` objects when processing
        :any:`Sourcefile` or :any:`Module` objects (default ``False``)
    recurse_to_internal_procedures : bool
        Apply transformation to all internal :any:`Subroutine` objects
        when processing :any:`Subroutine` objects (default ``False``)
    process_ignored_items : bool
        Apply transformation to "ignored" :any:`Item` objects for analysis.
        This might be needed if IPO-information needs to be passed across
        library boundaries.
    renames_items : bool
        Indicates to the :any:`Scheduler` that a transformation may change the name of
        the IR node corresponding to the processed :any:`Item` (e.g., by renaming
        a module or subroutine). The transformation has to take care of renaming
        processed the :any:`Item` itself but the :any:`Scheduler` will update its
        internal cache after the transformation has been applied (default ``False``).
    creates_items : bool
        Indicates to the :any:`Scheduler` that a transformation may create new
        scopes or other dependency nodes (e.g., by adding new routines to a
        module). The scheduler will run a discovery step after the transformation has
        been applied to include these new items in the dependency graph (default ``False``).
    """

    # Forces scheduler traversal in reverse order from the leaf nodes upwards
    reverse_traversal = False

    # Traverse a graph of Sourcefile options corresponding to scheduler items
    traverse_file_graph = False

    # Filter certain graph nodes to prune the graph and change connectivity
    item_filter = ProcedureItem  # This can also be a tuple of types

    # Recursion behaviour when invoking transformations via ``trafo.apply()``
    recurse_to_modules = False  # Recurse from Sourcefile to Module
    recurse_to_procedures = False  # Recurse from Sourcefile/Module to subroutines and functions
    recurse_to_internal_procedures = False  # Recurse to subroutines in ``contains`` clause

    # Option to process "ignored" items for analysis
    process_ignored_items = False

    # Control Scheduler cache update requirements after applying the transformation
    renames_items = False
    creates_items = False

    def __str__(self):
        """ Pretty-print transformation details """
        attrs = '\n    '.join(pformat(self.__dict__).splitlines())
        header = f'<{self.__class__.__name__}  [{self.__class__.__module__}]'
        return f'{header}\n    {attrs}>'

    def transform_subroutine(self, routine, **kwargs):
        """
        Defines the transformation to apply to :any:`Subroutine` items.

        For transformations that modify :any:`Subroutine` objects, this method
        should be implemented. It gets called via the dispatch method
        :meth:`apply`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

    def plan_subroutine(self, routine, **kwargs):
        """
        Define the planning steps to apply for :any:`Subroutine` items.

        For transformations that modify the dependencies of :data:`routine`
        (e.g., adding new procedure calls, inlining calls, renaming the interface)
        this should be implemented. It gets called via the dispatch method :meth:`apply`
        if the optional ``plan_mode`` argument is set to `True`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

    def transform_module(self, module, **kwargs):
        """
        Defines the transformation to apply to :any:`Module` items.

        For transformations that modify :any:`Module` objects, this method
        should be implemented. It gets called via the dispatch method
        :meth:`apply`.

        Parameters
        ----------
        module : :any:`Module`
            The module to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

    def plan_module(self, module, **kwargs):
        """
        Define the planning steps to apply for :any:`Module` items.

        For transformations that modify the dependencies or definitions of :data:`module`
        (e.g., renaming the module, adding new subroutines, adding or removing imports)
        this should be implemented. It gets called via the dispatch method :meth:`apply`
        if the optional ``plan_mode`` argument is set to `True`.

        Parameters
        ----------
        module : :any:`Module`
            The module to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

    def transform_file(self, sourcefile, **kwargs):
        """
        Defines the transformation to apply to :any:`Sourcefile` items.

        For transformations that modify :any:`Sourcefile` objects, this method
        should be implemented. It gets called via the dispatch method
        :meth:`apply`.

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The sourcefile to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

    def plan_file(self, sourcefile, **kwargs):
        """
        Define the planning steps to apply for :any:`Sourcefile` items.

        For transformations that modify the definitions or dependencies of :data:`sourcefile`
        this should be implemented. It gets called via the dispatch method :meth:`apply`
        if the optional ``plan_mode`` argument is set to `True`.

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The sourcefile to be transformed.
        **kwargs : optional
            Keyword arguments for the transformation.
        """

    def apply(self, source, post_apply_rescope_symbols=False, plan_mode=False, **kwargs):
        """
        Dispatch method to apply transformation to :data:`source`.

        It dispatches to one of the type-specific dispatch methods
        :meth:`apply_file`, :meth:`apply_module`, or :meth:`apply_subroutine`.

        Parameters
        ----------
        source : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine`
            The source item to transform.
        post_apply_rescope_symbols : bool, optional
            Call ``rescope_symbols`` on :data:`source` after applying the
            transformation to clean up any scoping issues.
        **kwargs : optional
            Keyword arguments that are passed on to the methods defining the
            actual transformation.
        """
        if isinstance(source, Sourcefile):
            self.apply_file(source, plan_mode=plan_mode, **kwargs)

        if isinstance(source, Subroutine):
            self.apply_subroutine(source, plan_mode=plan_mode, **kwargs)

        if isinstance(source, Module):
            self.apply_module(source, plan_mode=plan_mode, **kwargs)

        if not plan_mode:
            self.post_apply(source, rescope_symbols=post_apply_rescope_symbols)

    def apply_file(self, sourcefile, plan_mode=False, **kwargs):
        """
        Apply transformation to all items in :data:`sourcefile`.

        This calls :meth:`transform_file` or, if :data:`plan_mode` is enabled,
        calls :meth:`plan_file`.

        If the :attr:`recurse_to_modules` class property is set, it
        will also invoke :meth:`apply` on all :any:`Module` objects in
        this :any:`Sourcefile`. Likewise, if
        :attr:`recurse_to_procedures` is set, it will invoke
        :meth:`apply` on all free :any:`Subroutine` objects in this
        :any:`Sourcefile`.

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The file to transform.
        plan_mode : bool, optional
            If enabled, apply planning mode.
        **kwargs : optional
            Keyword arguments that are passed on to transformation methods.
        """
        if not isinstance(sourcefile, Sourcefile):
            raise TypeError('Transformation.apply_file can only be applied to Sourcefile object')

        item = kwargs.pop('item', None)
        items = kwargs.pop('items', None)
        role = kwargs.pop('role', None)
        targets = kwargs.pop('targets', None)

        # Apply file-level transformations
        try:
            if plan_mode:
                self.plan_file(sourcefile, item=item, role=role, targets=targets, items=items, **kwargs)
            else:
                if sourcefile._incomplete:
                    raise RuntimeError('Transformation.apply_file requires Sourcefile to be complete')

                self.transform_file(sourcefile, item=item, role=role, targets=targets, items=items, **kwargs)
        except Exception as e:
            raise TransformationError(
                message=f'Error in Sourcefile {sourcefile.path!s} -- {e!s}',
                transformation=type(self), source=sourcefile
            ) from e

        # Recurse to modules, if configured
        if self.recurse_to_modules:
            if items:
                # Recursion into all module items in the current file
                for item in items:
                    if isinstance(item, ModuleItem):
                        # Currently, we don't get the role for modules correct as 'driver'
                        # if the role overwrite in the config marks only specific procedures
                        # as driver, but everything else as kernel by default. This is in particular the
                        # case, if the ModuleWrapTransformation is applied to a driver routine.
                        # For that reason, we set the role as unspecified (None) if not the role is
                        # universally equal throughout the module
                        item_role = item.role
                        definitions_roles = {_it.role for _it in items if _it.scope_name == item.name}
                        if definitions_roles != {item_role}:
                            item_role = None

                        # Provide the list of items that belong to this module
                        item_items = tuple(_it for _it in items if _it.scope is item.ir)

                        try:
                            if plan_mode:
                                self.plan_module(
                                    item.ir, item=item, role=item_role, targets=item.targets, items=item_items, **kwargs
                                )
                            else:
                                self.transform_module(
                                    item.ir, item=item, role=item_role, targets=item.targets, items=item_items, **kwargs
                                )
                        except Exception as e:
                            raise TransformationError(
                                message=f'Error in Module {item.ir.name} -- {e!s}',
                                transformation=type(self), source=item.ir
                            ) from e
            else:
                for module in sourcefile.modules:
                    try:
                        if plan_mode:
                            self.plan_module(module, item=item, role=role, targets=targets, items=items, **kwargs)
                        else:
                            self.transform_module(module, item=item, role=role, targets=targets, items=items, **kwargs)
                    except Exception as e:
                        raise TransformationError(
                            message=f'Error in Module {module.name} -- {e!s}',
                            transformation=type(self), source=module
                        ) from e

        # Recurse into procedures, if configured
        if self.recurse_to_procedures:
            if items:
                # Recursion into all subroutine items in the current file
                for item in items:
                    if isinstance(item, ProcedureItem):
                        try:
                            if plan_mode:
                                self.plan_subroutine(
                                    item.ir, item=item, role=item.role, targets=item.targets, **kwargs
                                )
                            else:
                                self.transform_subroutine(
                                    item.ir, item=item, role=item.role, targets=item.targets, **kwargs
                                )
                        except Exception as e:
                            raise TransformationError(
                                message=f'Error in Procedure {item.ir.name} -- {e!s}',
                                transformation=type(self), source=item.ir
                            ) from e
            else:
                for routine in sourcefile.all_subroutines:
                    try:
                        if plan_mode:
                            self.plan_subroutine(routine, item=item, role=role, targets=targets, **kwargs)
                        else:
                            self.transform_subroutine(routine, item=item, role=role, targets=targets, **kwargs)
                    except Exception as e:
                        raise TransformationError(
                            message=f'Error in Procedure {routine.name} -- {e!s}',
                            transformation=type(self), source=routine
                        ) from e

    def apply_subroutine(self, subroutine, plan_mode=False, **kwargs):
        """
        Apply transformation to a given :any:`Subroutine` object and its members.

        This calls :meth:`transform_subroutine` or, if :data:`plan_mode` is enabled,
        calls :meth:`plan_subroutine`.

        If the :attr:`recurse_to_member_procedures` class property is
        set, it will also invoke :meth:`apply` on all
        :any:`Subroutine` objects in the ``contains`` clause of this
        :any:`Subroutine`.

        Parameters
        ----------
        subroutine : :any:`Subroutine`
            The subroutine to transform.
        plan_mode : bool, optional
            If enabled, apply planning mode.
        **kwargs : optional
            Keyword arguments that are passed on to transformation methods.
        """
        if not isinstance(subroutine, Subroutine):
            raise TypeError('Transformation.apply_subroutine can only be applied to Subroutine object')

        # Apply the actual transformation for subroutines
        try:
            if plan_mode:
                self.plan_subroutine(subroutine, **kwargs)
            else:
                if subroutine._incomplete:
                    raise RuntimeError('Transformation.apply_subroutine requires Subroutine to be complete')

                self.transform_subroutine(subroutine, **kwargs)
        except Exception as e:
            raise TransformationError(
                message=f'Error in Procedure {subroutine.name} -- {e!s}',
                transformation=type(self), source=subroutine
            ) from e

        # Recurse to internal procedures
        if self.recurse_to_internal_procedures:
            for routine in subroutine.subroutines:
                self.apply_subroutine(routine, plan_mode=plan_mode, **kwargs)

    def apply_module(self, module, plan_mode=False, **kwargs):
        """
        Apply transformation to a given :any:`Module` object and its members.

        This calls :meth:`transform_module` or, if :data:`plan_mode` is enabled,
        calls :meth:`plan_module`.

        If the :attr:`recurse_to_procedures` class property is set,
        it will also invoke :meth:`apply` on all :any:`Subroutine`
        objects in the ``contains`` clause of this :any:`Module`.

        Parameters
        ----------
        module : :any:`Module`
            The module to transform.
        plan_mode : bool, optional
            If enabled, apply planning mode.
        **kwargs : optional
            Keyword arguments that are passed on to transformation methods.
        """
        if not isinstance(module, Module):
            raise TypeError('Transformation.apply_module can only be applied to Module object')

        # Apply the actual transformation for modules
        try:
            if plan_mode:
                self.plan_module(module, **kwargs)
            else:
                if module._incomplete:
                    raise RuntimeError('Transformation.apply_module requires Module to be complete')

                self.transform_module(module, **kwargs)
        except Exception as e:
            raise TransformationError(
                message=f'Error in Module {module.name} -- {e!s}',
                transformation=type(self), source=module
            ) from e

        # Recurse to procedures contained in this module
        if self.recurse_to_procedures:
            for routine in module.subroutines:
                self.apply_subroutine(routine, plan_mode=plan_mode, **kwargs)

    def post_apply(self, source, rescope_symbols=False):
        """
        Dispatch method for actions to be carried out after applying a transformation
        to :data:`source`.

        It dispatches to one of the type-specific dispatch methods
        :meth:`post_apply_file`, :meth:`post_apply_module`, or :meth:`post_apply_subroutine`.

        Parameters
        ----------
        source : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine`
            The source item to transform.
        rescope_symbols : bool, optional
            Call ``rescope_symbols`` on :data:`source`
        """
        if isinstance(source, Sourcefile):
            self.post_apply_file(source, rescope_symbols)

        if isinstance(source, Subroutine):
            self.post_apply_subroutine(source, rescope_symbols)

        if isinstance(source, Module):
            self.post_apply_module(source, rescope_symbols)

    def post_apply_file(self, sourcefile, rescope_symbols):
        """
        Apply actions after applying a transformation to :data:`sourcefile`.

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The file to transform.
        rescope_symbols : bool
            Call ``rescope_symbols`` on modules and subroutines in :data:`sourcefile`
        """
        if not isinstance(sourcefile, Sourcefile):
            raise TypeError('Transformation.post_apply_file can only be applied to Sourcefile object')

        for module in sourcefile.modules:
            self.post_apply_module(module, rescope_symbols)

        for routine in sourcefile.subroutines:
            self.post_apply_subroutine(routine, rescope_symbols)


    def post_apply_subroutine(self, subroutine, rescope_symbols):
        """
        Apply actions after applying a transformation to :data:`subroutine`.

        Parameters
        ----------
        subroutine : :any:`Subroutine`
            The file to transform.
        rescope_symbols : bool
            Call ``rescope_symbols`` on :data:`subroutine`
        """
        if not isinstance(subroutine, Subroutine):
            raise TypeError('Transformation.post_apply_subroutine can only be applied to Subroutine object')

        for routine in subroutine.members:
            self.post_apply_subroutine(routine, False)

        # Ensure all objects in the IR are in the subroutine's or a parent scope.
        if rescope_symbols:
            subroutine.rescope_symbols()

    def post_apply_module(self, module, rescope_symbols):
        """
        Apply actions after applying a transformation to :data:`module`.

        Parameters
        ----------
        module : :any:`Module`
            The file to transform.
        rescope_symbols : bool
            Call ``rescope_symbols`` on :data:`module`
        """
        if not isinstance(module, Module):
            raise TypeError('Transformation.post_apply_module can only be applied to Module object')

        for routine in module.subroutines:
            self.post_apply_subroutine(routine, False)

        # Ensure all objects in the IR are in the module's scope.
        if rescope_symbols:
            module.rescope_symbols()
