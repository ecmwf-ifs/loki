from abc import ABCMeta, abstractmethod
from pathlib import Path

from loki.module import Module
from loki.sourcefile import SourceFile
from loki.backend import fgen
from loki.visitors import FindNodes
from loki.ir import CallStatement


__all__ = ['AbstractTransformation', 'BasicTransformation']


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

    def rename_routine(self, routine, **kwargs):
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

    def rename_calls(self, routine, **kwargs):
        """
        Update calls to actively transformed subroutines.
        """
        suffix = kwargs.get('suffix', None)
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.context is not None and call.context.active:
                call._update(name='%s_%s' % (call.name, suffix))

    def write_to_file(self, routine, **kwargs):
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
