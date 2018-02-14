from collections import Iterable

__all__ = ['flatten']


def flatten(l):
    """
    Flatten a hierarchy of nested lists into a plain list.
    """
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist
