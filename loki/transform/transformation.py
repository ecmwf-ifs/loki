from abc import ABCMeta, abstractmethod
from pathlib import Path

from loki.module import Module
from loki.sourcefile import SourceFile
from loki.subroutine import Subroutine
from loki.backend import fgen
from loki.visitors import FindNodes
from loki.ir import CallStatement


__all__ = ['Transformation', 'AbstractTransformation', 'BasicTransformation']


class Transformation:
    """
    Base class for source code transformations that manipulate source
    items like `Subroutine` or `Module` in place via the
    `item.apply(transform)` method.

    The source transformations to be applied should be defined in the
    following class-specific methods:
    * `transform_subroutine(self, routine, **kwargs)`
    * `transform_module(self, module, **kwargs)`
    * `transform_file(self, sourcefile, **kwargs)`

    The generic dispatch mechanism behind the `Transform.apply(source,
    **kwargs)` method will ensure that all hierarchies of the data
    model are traversed and apply the specific method for each
    level. Note that in `SourceFile` objects, `Module`s will be
    traversed before standalone `Subroutine` objects.
    """

    def transform_subroutine(self, routine, **kwargs):
        """
        Defines the transformation to apply to `Subroutine` items.
        """

    def transform_module(self, module, **kwargs):
        """
        Defines the transformation to apply to `Module` items.
        """

    def transform_file(self, module, **kwargs):
        """
        Defines the transformation to apply to `SourceFile` items.
        """

    def apply(self, source, **kwargs):
        """
        Apply transformation to all source items in :param source:.
        """
        if isinstance(source, SourceFile):
            self.apply_file(source, **kwargs)

        if isinstance(source, Subroutine):
            self.apply_subroutine(source, **kwargs)

        if isinstance(source, Module):
            self.apply_module(source, **kwargs)

    def apply_file(self, sourcefile, **kwargs):
        """
        Apply transformation to all items in :param sourcefile:.
        """
        if not isinstance(sourcefile, SourceFile):
            raise TypeError('Transformation.apply_file can only be applied to SourceFile object')

        for module in sourcefile.modules:
            self.apply(module, **kwargs)

        for routine in sourcefile.subroutines:
            self.apply(routine, **kwargs)

    def apply_subroutine(self, subroutine, **kwargs):
        """
        Apply transformation to a given `Subroutine` object.
        """
        if not isinstance(subroutine, Subroutine):
            raise TypeError('Transformation.apply_routine can only be applied to Subroutine object')

        # Apply the actual transformation for subroutines
        self.transform_subroutine(subroutine, **kwargs)

        # Recurse on subroutine members
        for member in subroutine.members:
            self.apply(member, **kwargs)

    def apply_module(self, module, **kwargs):
        """
        Apply transformation to a given `Module` object.
        """
        if not isinstance(module, Module):
            raise TypeError('Transformation.apply_module can only be applied to Module object')

        # Apply the actual transformation for modules
        self.transform_module(module, **kwargs)

        # Call the dispatch for all contained subroutines
        for routine in module.subroutines:
            self.apply(routine, **kwargs)


class AbstractTransformation:
    """
    Abstract base class that encapsulates the workflow of a single
    pre-defined source code transformation.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def _pipeline(self, source, **kwargs):
        return

    def apply(self, source, **kwargs):
        """
        Perform a source-to-source transformation pass on the provided
        :class:`Subroutine` or :class:`Module` according to the defined
        ``_pipeline`` and any given runtime parameters.

        :param source: :class:`Subroutine` or :class:`Module` to transform.
        :param kwargs: Parameter set to configure the transformation pass.
        """
        self._pipeline(source, **kwargs)


class BasicTransformation(AbstractTransformation):
    """
    Basic transformation encoding some simple housekeeping transformations,
    such as renaming or wrapping :class:`Subroutine` in :class:`Module`s.
    """

    def _pipeline(self, source, **kwargs):
        """
        Simple example pipeline that renames a :class:`Subroutine`,
        wraps it in a :class:`Module` object and writes the transformed
        module to file.
        """

        self.rename_routine(source, **kwargs)
        self.rename_calls(source, **kwargs)
        self.write_to_file(source, **kwargs)

    @staticmethod
    def rename_routine(routine, **kwargs):
        """
        Appends a suffix to :class:`Subroutine` names to distinguish
        them from the original version, and updates all subroutine
        :class:`CallStatement`s accordingly.

        :param suffix: The suffix to append to the subroutine name.
        """
        suffix = kwargs.get('suffix', None)
        if suffix is not None:
            # Rename the current subroutine
            routine.name += '_%s' % suffix

    @staticmethod
    def rename_calls(routine, **kwargs):
        """
        Update calls to actively transformed subroutines.
        """
        suffix = kwargs.get('suffix', None)
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.context is not None and call.context.active:
                call._update(name='%s_%s' % (call.name, suffix))

    @staticmethod
    def write_to_file(routine, **kwargs):
        """
        Does what it says on the tin.

        :param filename: The filename to be written to.
        """
        filename = kwargs.get('filename')
        module_wrap = kwargs.get('module_wrap', True)

        if module_wrap:
            name = '%s_MOD' % routine.name.upper()
            content = Module(name=name, routines=[routine])
        else:
            content = routine

        # Re-generate source code for content and write to file
        SourceFile.to_file(source=fgen(content), path=Path(filename))
