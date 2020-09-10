from loki.module import Module
from loki.sourcefile import SourceFile
from loki.subroutine import Subroutine
from loki.ir import CallStatement


__all__ = ['Transformation']


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

    def transform_file(self, sourcefile, **kwargs):
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

        # Apply file-level transformations
        self.transform_file(sourcefile, **kwargs)

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
