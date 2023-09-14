# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Base class definition for :ref:`transformations`.
"""
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.bulk.item import SubroutineItem


__all__ = ['Transformation']


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
    that all hierarchies of the data model are traversed and apply the specific
    method for each level.

    Note that in :any:`Sourcefile` objects, all :any:`Module` members will be
    traversed before standalone :any:`Subroutine` objects.
    """

    # Below is a set of flags that consfigure the corresponding Scheduler behaviour

    # Forces scheduler traversal in reverse order from the leaf nodes upwards
    reverse_traversal = False

    # The following options configure the iteration space of the
    # Scheduler traversal; it configures which graph-type is used and
    # which nodes are traversed when applying the Transformation.
    #
    # TODO: All these traversal options will eventually superseded by SGraph options
    traverse_file_graph = False

    item_filter = SubroutineItem  # This can also be a tuple of types

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

    def apply(self, source, post_apply_rescope_symbols=False, **kwargs):
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
            self.apply_file(source, **kwargs)

        if isinstance(source, Subroutine):
            self.apply_subroutine(source, **kwargs)

        if isinstance(source, Module):
            self.apply_module(source, **kwargs)

        self.post_apply(source, rescope_symbols=post_apply_rescope_symbols)

    def apply_file(self, sourcefile, **kwargs):
        """
        Apply transformation to all items in :data:`sourcefile`.

        This calls :meth:`transform_file`.

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The file to transform.
        **kwargs : optional
            Keyword arguments that are passed on to transformation methods.
        """
        if not isinstance(sourcefile, Sourcefile):
            raise TypeError('Transformation.apply_file can only be applied to Sourcefile object')

        if sourcefile._incomplete:
            raise RuntimeError('Transformation.apply_file requires Sourcefile to be complete')

        # Apply file-level transformations
        self.transform_file(sourcefile, **kwargs)

    def apply_subroutine(self, subroutine, **kwargs):
        """
        Apply transformation to a given :any:`Subroutine` object and its members.

        This calls :meth:`transform_subroutine`.

        Parameters
        ----------
        subroutine : :any:`Subroutine`
            The subroutine to transform.
        **kwargs : optional
            Keyword arguments that are passed on to transformation methods.
        """
        if not isinstance(subroutine, Subroutine):
            raise TypeError('Transformation.apply_subroutine can only be applied to Subroutine object')

        if subroutine._incomplete:
            raise RuntimeError('Transformation.apply_subroutine requires Subroutine to be complete')

        # Apply the actual transformation for subroutines
        self.transform_subroutine(subroutine, **kwargs)

    def apply_module(self, module, **kwargs):
        """
        Apply transformation to a given :any:`Module` object and its members.

        This calls :meth:`transform_module`.

        Parameters
        ----------
        module : :any:`Module`
            The module to transform.
        **kwargs : optional
            Keyword arguments that are passed on to transformation methods.
        """
        if not isinstance(module, Module):
            raise TypeError('Transformation.apply_module can only be applied to Module object')

        if module._incomplete:
            raise RuntimeError('Transformation.apply_module requires Module to be complete')

        # Apply the actual transformation for modules
        self.transform_module(module, **kwargs)

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
