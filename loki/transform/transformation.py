"""
Base class definition for :ref:`transformations`.
"""
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine


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

    def apply(self, source, **kwargs):
        """
        Dispatch method to apply transformation to all source items in
        :data:`source`.

        It dispatches to one of the type-specific dispatch methods
        :meth:`apply_file`, :meth:`apply_module`, or :meth:`apply_subroutine`.

        Parameters
        ----------
        source : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine`
            The source item to transform.
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

        self.post_apply(source)

    def apply_file(self, sourcefile, **kwargs):
        """
        Apply transformation to all items in :data:`sourcefile`.

        This calls :meth:`transform_file` and dispatches the transformation
        for all :any:`Module` and :any:`Subroutine` objects in the file.

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

        for module in sourcefile.modules:
            self.apply(module, **kwargs)

        for routine in sourcefile.subroutines:
            self.apply(routine, **kwargs)

    def apply_subroutine(self, subroutine, **kwargs):
        """
        Apply transformation to a given :any:`Subroutine` object and its members.

        This calls :meth:`transform_subroutine` and dispatches the transformation
        for all :any:`Subroutine` members.

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

        # Recurse on subroutine members
        for member in subroutine.members:
            self.apply(member, **kwargs)

    def apply_module(self, module, **kwargs):
        """
        Apply transformation to a given :any:`Module` object and its members.

        This calls :meth:`transform_module` and dispatches the transformation
        for all :any:`Subroutine` members.

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

        # Call the dispatch for all contained subroutines
        for routine in module.subroutines:
            self.apply(routine, **kwargs)

    def post_apply(self, source):
        """
        Dispatch method for actions to be carried out after applying a transformation
        to :data:`source`.

        It dispatches to one of the type-specific dispatch methods
        :meth:`post_apply_file`, :meth:`post_apply_module`, or :meth:`post_apply_subroutine`.

        Parameters
        ----------
        source : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine`
            The source item to transform.
        """
        if isinstance(source, Sourcefile):
            self.post_apply_file(source)

        if isinstance(source, Subroutine):
            self.post_apply_subroutine(source)

        if isinstance(source, Module):
            self.post_apply_module(source)

    def post_apply_file(self, sourcefile):
        """
        Apply actions after applying a transformation to :data:`sourcefile`.

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The file to transform.
        """
        if not isinstance(sourcefile, Sourcefile):
            raise TypeError('Transformation.post_apply_file can only be applied to Sourcefile object')

    def post_apply_subroutine(self, subroutine):
        """
        Apply actions after applying a transformation to :data:`subroutine`.

        Parameters
        ----------
        subroutine : :any:`Subroutine`
            The file to transform.
        """
        if not isinstance(subroutine, Subroutine):
            raise TypeError('Transformation.post_apply_subroutine can only be applied to Subroutine object')

        # Ensure all objects in the IR are in the subroutine's or a parent scope.
        subroutine.rescope_symbols()

    def post_apply_module(self, module):
        """
        Apply actions after applying a transformation to :data:`module`.

        Parameters
        ----------
        module : :any:`Module`
            The file to transform.
        """
        if not isinstance(module, Module):
            raise TypeError('Transformation.post_apply_module can only be applied to Module object')

        # Ensure all objects in the IR are in the module's scope.
        module.rescope_symbols()
