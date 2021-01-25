from loki.tools import flatten

__all__ = ['Binary']


class Binary:
    """
    A binary build target to generate executables.
    """

    def __init__(self, name, objs=None, libs=None):
        self.name = name
        self.objs = objs or []
        self.libs = libs or []

    def build(self, builder):

        # Trigger build for object dependencies
        for obj in flatten(self.objs):
            obj.build(builder=builder)

        # Trigger build for library dependencies
        for lib in flatten(self.libs):
            lib.build(builder=builder)

        # TODO: Link the final binary
