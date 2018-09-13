__all__ = ['Binary']


class Binary(object):
    """
    A binary build target to generate executables.
    """

    def __init__(self, name, objs=None, libs=None):
        self.name = name
        self.objs = objs
        self.libs = libs
